#include "spotter/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace spotter
{

float Pose::Degrees() const
{
    float cosine = (turn(0, 0) + turn(1, 1) + turn(2, 2) - 1) / 2;   // the angle of a rotation, from its trace
    return std::acos(std::clamp(cosine, -1.0f, 1.0f)) * 180 / (float)CV_PI;
}

Pose Kabsch(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to)
{
    cv::Point3d fromCentre(0, 0, 0), toCentre(0, 0, 0);
    for (size_t i = 0; i < from.size(); i++)
    {
        fromCentre += cv::Point3d(from[i]);
        toCentre += cv::Point3d(to[i]);
    }
    fromCentre *= 1.0 / from.size();
    toCentre *= 1.0 / to.size();
    cv::Matx33d together = cv::Matx33d::zeros();   // how the two sets vary together (cross-covariance)
    for (size_t i = 0; i < from.size(); i++)
    {
        cv::Point3d p = cv::Point3d(from[i]) - fromCentre, q = cv::Point3d(to[i]) - toCentre;
        together += cv::Matx33d(p.x * q.x, p.x * q.y, p.x * q.z, p.y * q.x, p.y * q.y, p.y * q.z, p.z * q.x, p.z * q.y, p.z * q.z);
    }
    cv::Mat s, u, vt;
    cv::SVD::compute(cv::Mat(together), s, u, vt);
    cv::Mat turn = vt.t() * u.t();
    if (cv::determinant(turn) < 0)   // a mirror image, not a turn: flip the weakest axis
    {
        vt.row(2) *= -1;
        turn = vt.t() * u.t();
    }
    Pose pose;
    pose.turn = cv::Matx33f(cv::Matx33d(turn));
    pose.shift = cv::Point3f(toCentre) - pose.turn * cv::Point3f(fromCentre);
    return pose;
}

cv::Matx33f TurnAbout(const cv::Vec3d& w)
{
    double angle = cv::norm(w);
    if (angle < 1e-12) return cv::Matx33f::eye();
    cv::Vec3d k = w / angle;
    double c = std::cos(angle), s = std::sin(angle);
    cv::Matx33d turn;
    for (int row = 0; row < 3; row++)
        for (int column = 0; column < 3; column++) turn(row, column) = (row == column ? c : 0) + (1 - c) * k[row] * k[column];
    turn(0, 1) -= s * k[2]; turn(0, 2) += s * k[1];
    turn(1, 0) += s * k[2]; turn(1, 2) -= s * k[0];
    turn(2, 0) -= s * k[1]; turn(2, 1) += s * k[0];
    return cv::Matx33f(turn);
}

bool PlaneThrough(const cv::Point3f& a, const cv::Point3f& b, const cv::Point3f& c, Plane& plane)
{
    cv::Point3f normal = (b - a).cross(c - a);
    float length = (float)cv::norm(normal);
    if (length < 1e-6f) return false;
    plane.normal = normal / length;
    plane.distance = -plane.normal.dot(a);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return true;
}

Plane FitPlane(const std::vector<cv::Point3f>& points)
{
    cv::Mat data((int)points.size(), 3, CV_32F, (void*)points.data());
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point3f mean(pca.mean.at<float>(0), pca.mean.at<float>(1), pca.mean.at<float>(2));
    Plane plane{ cv::Point3f(pca.eigenvectors.at<float>(2, 0), pca.eigenvectors.at<float>(2, 1), pca.eigenvectors.at<float>(2, 2)), 0 };
    plane.distance = -plane.normal.dot(mean);
    if (plane.distance < 0) { plane.normal = -plane.normal; plane.distance = -plane.distance; }
    return plane;
}

PointGrid ThinCloud(const std::vector<cv::Point3f>& cloud, int width, int step)
{
    PointGrid grid;
    grid.columns = width / step;
    grid.rows = (int)cloud.size() / width / step;
    grid.points.resize(grid.columns * grid.rows);
    for (int row = 0; row < grid.rows; row++)
        for (int column = 0; column < grid.columns; column++)
            grid.points[row * grid.columns + column] = cloud[row * step * width + column * step];
    return grid;
}

std::vector<PlanePiece> FindPlanes(const PointGrid& grid, const PlaneSearch& search, std::mt19937& random)
{
    const std::vector<cv::Point3f>& points = grid.points;
    const int reach = search.neighbourCells;
    const float tiltSin = std::sin(search.maxTiltDegrees * (float)CV_PI / 180), tiltCos = std::cos(search.maxTiltDegrees * (float)CV_PI / 180);
    const cv::Point3f cameraUp(0, -1, 0);   // the camera's Y axis points DOWN
    std::uniform_int_distribution<int> pickOffset(-reach, reach);
    auto Facing = [&](const Plane& plane)
    {
        float upness = std::abs(plane.normal.dot(cameraUp));
        if (search.facing == PlaneSearch::Facing::Upright) return upness <= tiltSin;
        if (search.facing == PlaneSearch::Facing::Level) return upness >= tiltCos;
        return true;
    };

    // free[i]: cell i has a reading and no plane has taken it yet.
    std::vector<bool> free(points.size());
    int readings = 0;
    for (size_t i = 0; i < points.size(); i++) readings += (free[i] = !std::isnan(points[i].z));

    std::vector<PlanePiece> pieces;
    std::vector<int> freeCells;
    for (int attempt = 0; attempt < search.maxPlanes; attempt++)
    {
        freeCells.clear();
        for (int i = 0; i < (int)points.size(); i++) if (free[i]) freeCells.push_back(i);
        if (freeCells.size() < 3) break;

        // RANSAC: three points near each other (neighbours usually share a surface), their plane,
        // and how many free points lie on it. Keep the best.
        std::uniform_int_distribution<size_t> pickCell(0, freeCells.size() - 1);
        Plane best;
        int bestCount = 0;
        for (int t = 0; t < search.tries; t++)
        {
            int a = freeCells[pickCell(random)], row = a / grid.columns, column = a % grid.columns;
            int bRow = row + pickOffset(random), bColumn = column + pickOffset(random);
            int cRow = row + pickOffset(random), cColumn = column + pickOffset(random);
            if (std::min({ bRow, bColumn, cRow, cColumn }) < 0 || std::max(bRow, cRow) >= grid.rows ||
                std::max(bColumn, cColumn) >= grid.columns) continue;
            int b = bRow * grid.columns + bColumn, c = cRow * grid.columns + cColumn;
            if (!free[b] || !free[c]) continue;
            Plane plane;
            if (!PlaneThrough(points[a], points[b], points[c], plane) || !Facing(plane)) continue;
            int count = 0;
            for (int i : freeCells) count += std::abs(plane.DistanceTo(points[i])) < search.thicknessMeters;
            if (count > bestCount) { bestCount = count; best = plane; }
        }
        if (bestCount < search.minShare * readings) break;   // no big plane left

        // A plane runs on for ever, so it also slices strips off other surfaces. The surface itself
        // is one connected piece in the picture: keep the biggest.
        cv::Mat onPlane(grid.rows, grid.columns, CV_8U, cv::Scalar(0)), labels, stats, centroids;
        for (int i : freeCells)
            if (std::abs(best.DistanceTo(points[i])) < search.thicknessMeters) onPlane.data[i] = 255;
        int count = cv::connectedComponentsWithStats(onPlane, labels, stats, centroids);
        int biggest = 1;
        for (int piece = 2; piece < count; piece++)
            if (stats.at<int>(piece, cv::CC_STAT_AREA) > stats.at<int>(biggest, cv::CC_STAT_AREA)) biggest = piece;
        if (count < 2 || stats.at<int>(biggest, cv::CC_STAT_AREA) < search.minShare * readings)
        {
            // Too scattered to be one surface. Take its points away anyway, or the next search
            // finds the same plane again (the bug in 06b, 06c and 09c).
            for (int i : freeCells) if (onPlane.data[i]) free[i] = false;
            continue;
        }
        PlanePiece piece;
        std::vector<cv::Point3f> inliers;
        for (int i : freeCells)
        {
            if (labels.ptr<int>()[i] != biggest) continue;
            piece.cells.push_back(i);
            inliers.push_back(points[i]);
            free[i] = false;
        }
        piece.plane = FitPlane(inliers);
        pieces.push_back(piece);
    }
    return pieces;
}

}
