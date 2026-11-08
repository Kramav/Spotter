#include "spotter/register.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/flann.hpp>

namespace spotter
{

SurfacePoints SampleSurface(const std::vector<cv::Point3f>& cloud, int width, int step, int reach, float maxJumpShare)
{
    SurfacePoints surface;
    const int height = (int)cloud.size() / width, k = reach;
    for (int row = k; row + k < height; row += step)
        for (int column = k; column + k < width; column += step)
        {
            const cv::Point3f& p = cloud[row * width + column];
            const cv::Point3f &left = cloud[row * width + column - k], &right = cloud[row * width + column + k];
            const cv::Point3f &up = cloud[(row - k) * width + column], &down = cloud[(row + k) * width + column];
            if (std::isnan(p.z) || std::isnan(left.z) || std::isnan(right.z) || std::isnan(up.z) || std::isnan(down.z)) continue;
            if (std::abs(right.z - left.z) > maxJumpShare * p.z || std::abs(down.z - up.z) > maxJumpShare * p.z) continue;   // an edge
            cv::Point3f n = (right - left).cross(down - up);   // across x down = the facing
            float length = (float)cv::norm(n);
            if (!(length > 1e-9f)) continue;   // bug 2: the neighbours in a line give no facing
            n *= 1.0f / length;
            if (n.dot(p) > 0) n = -n;   // face the camera
            surface.points.push_back(p);
            surface.normals.push_back(n);
        }
    return surface;
}

IcpResult Icp(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to, const std::vector<cv::Point3f>& toNormals,
              const Pose& start, const IcpOptions& options)
{
    IcpResult result;
    result.pose = start;
    if (from.size() < 10 || to.size() < 10) return result;
    cv::Mat toData((int)to.size(), 3, CV_32F, (void*)to.data());
    cv::flann::Index tree(toData, cv::flann::KDTreeIndexParams(4));
    cv::Mat query((int)from.size(), 3, CV_32F), nearestIndex, squaredDistance;
    Pose& pose = result.pose;
    for (int round = 0; round < options.rounds; round++)
    {
        for (int i = 0; i < (int)from.size(); i++)
        {
            cv::Point3f p = pose(from[i]);
            query.at<float>(i, 0) = p.x;
            query.at<float>(i, 1) = p.y;
            query.at<float>(i, 2) = p.z;
        }
        tree.knnSearch(query, nearestIndex, squaredDistance, 1, cv::flann::SearchParams(32));
        const float maxPair = std::max(options.finalPairMeters, options.startPairMeters * std::pow(0.7f, (float)round));
        const float maxNeighbour = std::max(maxPair, 0.06f);   // the nearest sample may be a sample spacing away
        const bool last = maxPair == options.finalPairMeters;

        // The pairs: the moved point, the target's normal, and the gap along that normal.
        std::vector<cv::Point3f> ps, normals;
        std::vector<float> gaps;
        cv::Point3f centre(0, 0, 0);
        for (int i = 0; i < (int)from.size(); i++)
        {
            if (squaredDistance.at<float>(i) > maxNeighbour * maxNeighbour) continue;
            cv::Point3f p(query.at<float>(i, 0), query.at<float>(i, 1), query.at<float>(i, 2));
            int j = nearestIndex.at<int>(i);
            float gap = (p - to[j]).dot(toNormals[j]);
            if (!(std::abs(gap) <= maxPair)) continue;   // bug 2: written so a NaN gap is left out too
            ps.push_back(p);
            normals.push_back(toNormals[j]);
            gaps.push_back(gap);
            centre += p;
        }
        const int pairs = (int)ps.size();
        if (pairs < 10) break;
        centre *= 1.0f / pairs;
        double radius = 0;   // bug 13: the pairs' typical distance from their centre, to put turns in metres
        for (const cv::Point3f& p : ps) radius += (p - centre).dot(p - centre);
        radius = std::max(std::sqrt(radius / pairs), 1e-3);

        // Each pair: gap + turn . ((p - centre) x n) + shift . n = 0, for a small turn (axis x angle)
        // about the centre. The turn's columns are scaled by the radius (an angle there moves a
        // point radius x angle metres), so all 6 directions compare like with like. A x = b.
        cv::Matx66d A = cv::Matx66d::zeros();
        cv::Matx61d b = cv::Matx61d::zeros();
        double sum = 0;
        for (int i = 0; i < pairs; i++)
        {
            const cv::Point3f& n = normals[i];
            cv::Point3f c = (ps[i] - centre).cross(n);
            const double row[6] = { c.x / radius, c.y / radius, c.z / radius, n.x, n.y, n.z };
            for (int r = 0; r < 6; r++)
            {
                for (int k = 0; k < 6; k++) A(r, k) += row[r] * row[k];
                b(r) -= row[r] * gaps[i];
            }
            sum += gaps[i] * gaps[i];
        }
        // Solve direction by direction (eigenvectors): how strongly the pairs pin a direction is its
        // eigenvalue, about (share of pairs whose surface faces that way) x pairs.
        cv::Mat values, directions;
        cv::eigen(cv::Mat(A), values, directions);
        cv::Matx61d x = cv::Matx61d::zeros();
        int weak = 0;
        for (int k = 0; k < 6; k++)
        {
            double strength = values.at<double>(k);
            if (strength < options.weakShare * pairs) { weak++; continue; }
            double along = 0;
            for (int r = 0; r < 6; r++) along += directions.at<double>(k, r) * b(r);
            for (int r = 0; r < 6; r++) x(r) += along / strength * directions.at<double>(k, r);
        }
        if (last)
        {
            result.pairedShare = (float)pairs / from.size();
            result.rmsMeters = (float)std::sqrt(sum / pairs);
            result.weakDirections = weak;
        }
        // The step: turn about the centre, then shift: p -> turn (p - centre) + centre + shift.
        cv::Matx33f turn = TurnAbout(cv::Vec3d(x(0), x(1), x(2)) / radius);
        cv::Point3f shift((float)x(3), (float)x(4), (float)x(5));
        Pose step{ turn, centre - turn * centre + shift };
        pose = pose.Then(step);
        if (last && cv::norm(shift) < 1e-4 && step.Degrees() < 0.01f) break;   // settled
    }
    return result;
}

}
