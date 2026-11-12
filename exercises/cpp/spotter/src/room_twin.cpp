#include "spotter/room_twin.hpp"
#include "spotter/register.hpp"
#include "spotter/room.hpp"
#include "spotter/room_folder.hpp"
#include "spotter/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <opencv2/imgproc.hpp>

namespace spotter
{

std::filesystem::path NewestRoomFolder(const std::filesystem::path& root)
{
    std::filesystem::path newest;
    std::string newestSaved;
    for (const SavedRoom& room : FindRooms(root))
        if (newest.empty() || room.saved > newestSaved) { newest = room.folder; newestSaved = room.saved; }   // "YYYY-MM-DD HH:MM" sorts as text
    return newest;
}

bool LoadTwinRoom(const std::filesystem::path& folder, float cubeMeters, TwinRoom& room, std::string& problem)
{
    RoomInfo info;
    if (!ReadRoomInfo(folder, info, problem)) return false;
    std::vector<Scan> scans;
    std::string why;
    for (const std::filesystem::path& scanFolder : ScanFolders(folder))
    {
        Scan scan;
        int number;
        if (!LoadScan(scanFolder, info, scan, number, why)) std::printf("  %s skipped: %s\n", scanFolder.filename().string().c_str(), why.c_str());
        else if (!scan.excluded) scans.push_back(std::move(scan));
    }
    if (scans.empty()) { problem = "no scans to load in " + folder.string() + (why.empty() ? "" : " (the last skipped: " + why + ")"); return false; }
    std::vector<const Scan*> all;
    for (const Scan& s : scans) all.push_back(&s);
    RoomOptions options;
    options.voxelMeters = cubeMeters;
    float voxel;
    Mesh seen = FuseScans(all, info.depthCamera, options, voxel);
    room = TwinRoom();
    if (info.finished) room.firstCamera = info.firstCamera;
    else
    {
        std::mt19937 random(1);
        room.firstCamera = BuildRoom(all, seen, random, options).firstCamera;
    }
    // Each corner faces the sum of its triangles' facings (13's ICP measures along it).
    std::vector<cv::Point3f> facing(seen.points.size(), cv::Point3f(0, 0, 0));
    for (const auto& [name, triangles] : seen.groups)
        for (const cv::Vec3i& t : triangles)
        {
            cv::Point3f n = (seen.points[t[1]] - seen.points[t[0]]).cross(seen.points[t[2]] - seen.points[t[0]]);
            for (int k = 0; k < 3; k++) facing[t[k]] += n;
        }
    for (size_t i = 0; i < seen.points.size(); i++)
    {
        double length = cv::norm(facing[i]);
        if (!(length > 0)) continue;
        room.points.push_back(room.firstCamera * seen.points[i]);
        room.normals.push_back(room.firstCamera * (facing[i] * (float)(1 / length)));
        room.colours.push_back(seen.colours[i]);
    }
    room.from = cv::format("%s (%zu scans, %.0f cm cubes)", folder.filename().string().c_str(), scans.size(), voxel * 100);
    return !room.points.empty();
}

TwinRoom OneViewRoom(const std::vector<cv::Point3f>& cloud, const cv::Mat& colours, int width, float levelDegrees)
{
    // Level about the Kinect's left-right axis, then flip Y and Z (09b's flip): X right, Y up, Z back.
    const float c = std::cos(levelDegrees * (float)CV_PI / 180), s = std::sin(levelDegrees * (float)CV_PI / 180);
    TwinRoom room;
    room.firstCamera = cv::Matx33f(1, 0, 0, 0, -1, 0, 0, 0, -1) * cv::Matx33f(1, 0, 0, 0, c, -s, 0, s, c);
    room.from = "the Kinect's own view";
    const int step = 2, height = (int)cloud.size() / width;
    const float maxJumpShare = 0.03f;   // 09b: neighbours further apart than 3% of the distance = an edge
    for (int row = 0; row + step < height; row++)
        for (int column = 0; column + step < width; column++)
        {
            const cv::Point3f &p = cloud[row * width + column], &right = cloud[row * width + column + step], &below = cloud[(row + step) * width + column];
            if (std::isnan(p.z) || std::isnan(right.z) || std::isnan(below.z)) continue;
            if (std::abs(right.z - p.z) > maxJumpShare * p.z || std::abs(below.z - p.z) > maxJumpShare * p.z) continue;
            cv::Point3f n = (right - p).cross(below - p);
            double length = cv::norm(n);
            if (!(length > 0)) continue;
            room.points.push_back(room.firstCamera * p);
            room.normals.push_back(room.firstCamera * (n * (float)(1 / length)));
            room.colours.push_back(colours.at<cv::Vec3b>(row * width + column));
        }
    return room;
}

TwinFloor FindTwinFloor(const TwinRoom& room, float minAreaMeters2, float aboveFootMeters)
{
    const float facingDegrees = 20, stepMeters = 0.02f, gatherMeters = 0.04f, cellMeters = 0.05f, reportArea = 0.05f;
    const float levelCos = std::cos(facingDegrees * (float)CV_PI / 180), uprightSin = std::sin(facingDegrees * (float)CV_PI / 180);
    TwinFloor floor;
    std::vector<float> all, uprightHeights;
    std::vector<int> level;
    float lo = 1e9f, hi = -1e9f;
    for (int i = 0; i < (int)room.points.size(); i++)
    {
        const float y = room.points[i].y, ny = std::abs(room.normals[i].y);
        all.push_back(y);
        if (ny > levelCos) { level.push_back(i); lo = std::min(lo, y); hi = std::max(hi, y); }
        else if (ny < uprightSin) uprightHeights.push_back(y);
    }
    const float lowest = all.empty() ? 0 : Percentile(all, 0.02);   // legacy Demo 4's floor, for the console

    // The walls' foot: upright points per 2 cm of height; the lowest step holding a third of what a
    // typical step up the walls holds. Stray readings under the floor are far sparser.
    float foot = 1e9f;
    if (!uprightHeights.empty())
    {
        const float bottom = Percentile(uprightHeights, 0.0), top = Percentile(uprightHeights, 0.98);
        std::vector<int> steps((size_t)((top - bottom) / stepMeters) + 1, 0);
        for (float h : uprightHeights) if (h <= top) steps[std::min(steps.size() - 1, (size_t)((h - bottom) / stepMeters))]++;
        std::vector<float> filled;
        for (int count : steps) if (count > 0) filled.push_back((float)count);
        const float typical = Percentile(filled, 0.5);
        for (size_t k = 0; k < steps.size(); k++)
            if (steps[k] >= typical / 3) { foot = bottom + (k + 0.5f) * stepMeters; break; }
    }

    // Level heights in 2 cm bins: every peak's height and footprint, from the bottom up.
    const int bins = level.empty() ? 0 : std::min(100000, (int)((hi - lo) / stepMeters) + 1);
    std::vector<int> histogram(bins, 0);
    for (int i : level) histogram[std::min(bins - 1, (int)((room.points[i].y - lo) / stepMeters))]++;
    struct Piece { float y, area; bool tooHigh, tooLow; };
    std::vector<Piece> pieces;
    for (int bin = 0; bin < bins; bin++)
    {
        int before = bin > 0 ? histogram[bin - 1] : 0, after = bin + 1 < bins ? histogram[bin + 1] : 0;
        if (histogram[bin] < 10 || histogram[bin] <= before || histogram[bin] < after) continue;
        const float centre = lo + (bin + 0.5f) * stepMeters;
        std::vector<float> ys;
        std::vector<cv::Point2f> spots;
        float x0 = 1e9f, z0 = 1e9f, x1 = -1e9f, z1 = -1e9f;
        for (int i : level)
        {
            const cv::Point3f& p = room.points[i];
            if (std::abs(p.y - centre) >= gatherMeters) continue;
            ys.push_back(p.y);
            spots.emplace_back(p.x, p.z);
            x0 = std::min(x0, p.x); x1 = std::max(x1, p.x);
            z0 = std::min(z0, p.z); z1 = std::max(z1, p.z);
        }
        // Its footprint: the biggest connected piece of 5 cm cells it covers, seen from above.
        cv::Mat covered((int)((z1 - z0) / cellMeters) + 1, (int)((x1 - x0) / cellMeters) + 1, CV_8U, cv::Scalar(0));
        for (const cv::Point2f& s : spots) covered.at<uchar>((int)((s.y - z0) / cellMeters), (int)((s.x - x0) / cellMeters)) = 255;
        cv::Mat labels, stats, centroids;
        int count = cv::connectedComponentsWithStats(covered, labels, stats, centroids, 8);
        int biggest = 0;
        for (int k = 1; k < count; k++) biggest = std::max(biggest, stats.at<int>(k, cv::CC_STAT_AREA));
        const float y = Percentile(ys, 0.5);
        pieces.push_back({ y, biggest * cellMeters * cellMeters, foot < 1e9f && y > foot + aboveFootMeters, foot < 1e9f && y < foot - aboveFootMeters });
    }

    // The floor: the lowest big-enough piece near the walls' foot. A piece further below is a
    // reflection (the review's bug 7), unless it's the biggest level piece that isn't too high: a
    // reflection is never bigger than the floor it mirrors, so then the foot was measured too high
    // (the walls' lower part hidden behind furniture, or seen by few scans).
    float biggestArea = 0, floorArea = 0;
    for (const Piece& p : pieces) if (!p.tooHigh) biggestArea = std::max(biggestArea, p.area);
    for (const Piece& p : pieces)
    {
        const bool biggestBelow = p.tooLow && p.area >= biggestArea;
        floor.seen = p.area >= minAreaMeters2 && !p.tooHigh && (!p.tooLow || biggestBelow);
        if (floor.seen)
        {
            floor.y = p.y;
            floorArea = p.area;
            if (biggestBelow)
                floor.report.push_back(cv::format("  level readings at %.2f m: %.2f m^2 in one piece, %.2f m below the walls' foot, but the biggest level piece: the floor",
                                                  p.y, p.area, foot - p.y));
            break;
        }
        if (p.area >= reportArea && floor.report.size() < 8)
            floor.report.push_back(p.tooHigh ? cv::format("  level readings at %.2f m: %.2f m^2 in one piece, %.2f m above the walls' foot: a table top or the ceiling", p.y, p.area, p.y - foot)
                                 : p.tooLow  ? cv::format("  level readings at %.2f m: %.2f m^2 in one piece, %.2f m below the walls' foot: a reflection", p.y, p.area, foot - p.y)
                                             : cv::format("  level readings at %.2f m: %.2f m^2 in one piece, too small for the floor", p.y, p.area));
    }
    if (!floor.seen) floor.y = foot < 1e9f ? foot : lowest;
    floor.report.insert(floor.report.begin(), floor.seen
        ? cv::format("Floor: seen at %.2f m (%.2f m^2 of level readings in one piece); the walls' foot at %.2f m; the lowest 2%% of the points is at %.2f m",
                     floor.y, floorArea, foot < 1e9f ? foot : NAN, lowest)
        : cv::format("Floor: never seen (no level piece of %.2f m^2 near the walls' foot): guessed at the walls' foot, %.2f m; the lowest 2%% is at %.2f m",
                     minAreaMeters2, floor.y, lowest));
    return floor;
}

Located LocateKinect(const std::vector<cv::Point3f>& view, const TwinRoom& room, const LocateOptions& options)
{
    Located located;
    std::vector<cv::Point3f> part;
    for (size_t i = 0; i < view.size(); i += 4) part.push_back(view[i]);
    const IcpOptions icp{ 30, options.startPairMeters, 0.02f, 0.05f };
    // Nearest first: scan 1's own spot, then the shifts, then further turns.
    // ponytail: each Icp rebuilds its k-d tree over the room (legacy built it once); a tree-taking Icp if start-up gets slow.
    const cv::Point2i shifts[] = { { 0, 0 }, { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { 1, -1 }, { -1, -1 } };
    std::vector<IcpResult> tried;
    for (float degrees : options.turnsDegrees)
    {
        for (const cv::Point2i& shift : shifts)
        {
            Pose start{ TurnAbout(cv::Vec3d(0, degrees * CV_PI / 180, 0)) * room.firstCamera,
                        cv::Point3f(shift.x * options.shiftMeters, 0, shift.y * options.shiftMeters) };
            tried.push_back(Icp(part, room.points, room.normals, start, icp));
            if (tried.back().pairedShare >= options.enoughShare) break;
        }
        if (tried.back().pairedShare >= options.enoughShare) break;
    }
    std::sort(tried.begin(), tried.end(), [](const IcpResult& a, const IcpResult& b) { return a.pairedShare > b.pairedShare; });
    IcpResult best = Icp(view, room.points, room.normals, tried[0].pose, icp);
    located.pose = best.pose;
    located.share = best.pairedShare;
    located.tried = (int)tried.size();

    // The best distinct fits: the turn about up and the shift seen from above, from scan 1's spot.
    auto Turned = [&](const Pose& p)
    {
        cv::Matx33f r = p.turn * room.firstCamera.t();
        return std::atan2(r(0, 2), r(0, 0)) * 180 / (float)CV_PI;
    };
    std::vector<const IcpResult*> listed;
    for (const IcpResult& fit : tried)
    {
        bool same = false;
        for (const IcpResult* other : listed)
            same = same || (cv::norm(fit.pose.shift - other->pose.shift) < 0.1 && std::abs(Turned(fit.pose) - Turned(other->pose)) < 3);
        if (same) continue;
        if (listed.size() == 1 && fit.pairedShare > tried[0].pairedShare - 0.05f &&
            (cv::norm(fit.pose.shift - listed[0]->pose.shift) > 0.3 || std::abs(Turned(fit.pose) - Turned(listed[0]->pose)) > 10))
            located.twoAlike = true;
        located.report += cv::format("  %3.0f%%  turned %+4.0f deg, moved (%+.2f, %+.2f) m from scan 1's spot\n", fit.pairedShare * 100,
                                     Turned(fit.pose), fit.pose.shift.x, fit.pose.shift.z);
        listed.push_back(&fit);
        if (listed.size() == 3) break;
    }
    return located;
}

}
