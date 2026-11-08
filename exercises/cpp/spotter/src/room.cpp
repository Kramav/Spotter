#include "spotter/room.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <opencv2/imgproc.hpp>

#include "spotter/stats.hpp"

namespace spotter
{

Mesh FuseScans(const std::vector<const Scan*>& scans, const PinholeCamera& lens, const RoomOptions& options, float& voxel)
{
    Mesh mesh;
    voxel = options.voxelMeters;
    std::vector<float> xs, ys, zs;   // the box to fill: around the scans' points, 1-99% so strays don't blow it up
    for (const Scan* scan : scans)
        for (const cv::Point3f& p : scan->surface.points)
        {
            cv::Point3f q = scan->pose(p);
            xs.push_back(q.x); ys.push_back(q.y); zs.push_back(q.z);
        }
    if (xs.empty()) return mesh;
    const float margin = 0.3f;
    cv::Point3f lo(Percentile(xs, 0.01) - margin, Percentile(ys, 0.01) - margin, Percentile(zs, 0.01) - margin);
    cv::Point3f hi(Percentile(xs, 0.99) + margin, Percentile(ys, 0.99) + margin, Percentile(zs, 0.99) + margin);
    int nx, ny, nz;
    while (true)
    {
        nx = (int)std::ceil((hi.x - lo.x) / voxel);
        ny = (int)std::ceil((hi.y - lo.y) / voxel);
        nz = (int)std::ceil((hi.z - lo.z) / voxel);
        if ((double)nx * ny * nz <= options.maxVoxels) break;
        voxel *= 1.25f;   // too big a room for this cube size: coarser cubes
    }
    const size_t count = (size_t)nx * ny * nz;
    std::vector<float> distance(count, 0), weight(count, 0), colourWeight(count, 0);
    std::vector<cv::Vec3f> colourSum(count, cv::Vec3f(0, 0, 0));
    const float truncation = options.truncationVoxels * voxel;
    auto index = [&](int x, int y, int z) { return ((size_t)z * ny + y) * nx + x; };

    for (const Scan* scan : scans)
    {
        const cv::Matx33f back = scan->pose.turn.t();   // room -> this scan's camera
        cv::parallel_for_(cv::Range(0, nz), [&](const cv::Range& slices)
        {
            for (int z = slices.start; z < slices.end; z++)
                for (int y = 0; y < ny; y++)
                    for (int x = 0; x < nx; x++)
                    {
                        cv::Point3f centre(lo.x + (x + 0.5f) * voxel, lo.y + (y + 0.5f) * voxel, lo.z + (z + 0.5f) * voxel);
                        cv::Point3f p = back * (centre - scan->pose.shift);
                        if (p.z < 0.3f) continue;   // behind or right at the camera
                        int column = (int)std::lround(p.x * lens.fx / p.z + lens.cx - 0.5f);   // PinholeCamera::Project, inline
                        int row = (int)std::lround(p.y * lens.fy / p.z + lens.cy - 0.5f);
                        if (column < 0 || row < 0 || column >= lens.width || row >= lens.height) continue;
                        float seen = scan->cloud[row * lens.width + column].z;
                        if (std::isnan(seen)) continue;
                        float signedDistance = seen - p.z;            // + = in front of what the camera saw
                        if (signedDistance < -truncation) continue;   // hidden behind it: unknown
                        size_t i = index(x, y, z);
                        float value = std::min(1.0f, signedDistance / truncation);
                        distance[i] = (distance[i] * weight[i] + value) / (weight[i] + 1);
                        weight[i] += 1;
                        if (signedDistance < truncation)
                        {
                            cv::Vec3b c = scan->picture.at<cv::Vec3b>(row, column);
                            if (c != cv::Vec3b(0, 0, 0)) { colourSum[i] += cv::Vec3f(c[0], c[1], c[2]); colourWeight[i] += 1; }
                        }
                    }
        });
    }

    // The surface net: each cube of 8 neighbouring voxels where the sign changes gets one point, at
    // the average of where the surface crosses its edges; every voxel edge that crosses the surface
    // joins the 4 cubes around it into a square (two triangles).
    std::vector<int> pointOf(count, -1);
    auto corner = [](int c) { return cv::Point3f((float)(c & 1), (float)((c >> 1) & 1), (float)((c >> 2) & 1)); };
    for (int z = 0; z + 1 < nz; z++)
        for (int y = 0; y + 1 < ny; y++)
            for (int x = 0; x + 1 < nx; x++)
            {
                float value[8];
                int inside = 0;
                bool seenAll = true;
                cv::Vec3f colour(0, 0, 0);
                float colourCount = 0;
                for (int c = 0; c < 8 && seenAll; c++)
                {
                    size_t i = index(x + (c & 1), y + ((c >> 1) & 1), z + ((c >> 2) & 1));
                    seenAll = weight[i] > 0;
                    value[c] = distance[i];
                    if (value[c] < 0) inside++;
                    colour += colourSum[i];
                    colourCount += colourWeight[i];
                }
                if (!seenAll || inside == 0 || inside == 8) continue;   // never seen, or no surface here
                cv::Point3f sum(0, 0, 0);
                int crossings = 0;
                for (int bit : { 1, 2, 4 })
                    for (int c = 0; c < 8; c++)
                    {
                        if (c & bit) continue;
                        float a = value[c], b = value[c | bit];
                        if ((a < 0) == (b < 0)) continue;
                        sum += corner(c) + (corner(c | bit) - corner(c)) * (a / (a - b));   // where it crosses zero
                        crossings++;
                    }
                cv::Point3f inCube = sum * (1.0f / crossings);
                pointOf[index(x, y, z)] = (int)mesh.points.size();
                mesh.points.emplace_back(lo.x + (x + 0.5f + inCube.x) * voxel, lo.y + (y + 0.5f + inCube.y) * voxel,
                                         lo.z + (z + 0.5f + inCube.z) * voxel);
                mesh.colours.push_back(colourCount > 0 ? cv::Vec3b(colour / colourCount) : cv::Vec3b(128, 128, 128));
            }
    std::vector<cv::Vec3i>& faces = mesh.groups["seen_room"];
    const int size[3] = { nx, ny, nz };
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++)
            {
                size_t i = index(x, y, z);
                if (weight[i] == 0) continue;
                for (int axis = 0; axis < 3; axis++)
                {
                    int at[3] = { x, y, z };
                    if (at[axis] + 1 >= size[axis]) continue;
                    at[axis]++;
                    size_t j = index(at[0], at[1], at[2]);
                    if (weight[j] == 0 || (distance[i] < 0) == (distance[j] < 0)) continue;
                    int u = (axis + 1) % 3, w = (axis + 2) % 3, quad[4];   // the 4 cubes around this edge
                    const int du[4] = { -1, 0, 0, -1 }, dw[4] = { -1, -1, 0, 0 };
                    bool complete = true;
                    for (int k = 0; k < 4 && complete; k++)
                    {
                        int c[3] = { x, y, z };
                        c[u] += du[k];
                        c[w] += dw[k];
                        complete = c[u] >= 0 && c[w] >= 0 && c[u] + 1 < size[u] && c[w] + 1 < size[w];
                        if (complete) { quad[k] = pointOf[index(c[0], c[1], c[2])]; complete = quad[k] >= 0; }
                    }
                    if (!complete) continue;
                    float facingArray[3] = { 0, 0, 0 };   // face the empty (+) side
                    facingArray[axis] = distance[i] > 0 ? -1.0f : 1.0f;
                    cv::Point3f facing(facingArray[0], facingArray[1], facingArray[2]);
                    cv::Point3f normal = (mesh.points[quad[1]] - mesh.points[quad[0]]).cross(mesh.points[quad[2]] - mesh.points[quad[0]]);
                    if (normal.dot(facing) < 0) std::swap(quad[1], quad[3]);
                    faces.emplace_back(quad[0], quad[1], quad[2]);
                    faces.emplace_back(quad[0], quad[2], quad[3]);
                }
            }
    return mesh;
}

Room BuildRoom(const std::vector<const Scan*>& scans, const Mesh& seen, std::mt19937& random, const RoomOptions& options)
{
    Room room;
    room.mesh = seen;
    if (scans.empty()) return room;
    const int width = scans[0]->picture.cols, height = scans[0]->picture.rows, step = options.sampleStep;
    const int gridColumns = width / step, gridRows = height / step, cells = gridColumns * gridRows;   // as ThinCloud
    const cv::Point3f none(NAN, NAN, NAN);
    struct RoomPiece { Plane plane; std::vector<cv::Point3f> points; cv::Vec3d colourSum; std::vector<std::pair<int, int>> cells; };
    std::vector<RoomPiece> pieces;
    std::vector<std::vector<cv::Point3f>> roomGrid(scans.size());   // each scan's sampled points, in the room (first camera)
    std::vector<std::vector<cv::Vec3b>> gridColour(scans.size());
    std::vector<std::vector<float>> gridDepth(scans.size());          // ...and their distance from their own camera

    for (size_t s = 0; s < scans.size(); s++)
    {
        PointGrid grid = ThinCloud(scans[s]->cloud, width, step);
        roomGrid[s].resize(cells);
        gridColour[s].resize(cells);
        gridDepth[s].resize(cells);
        for (int cell = 0; cell < cells; cell++)
        {
            int pixel = (cell / gridColumns) * step * width + (cell % gridColumns) * step;
            const cv::Point3f& p = grid.points[cell];
            roomGrid[s][cell] = std::isnan(p.z) ? none : scans[s]->pose(p);
            gridColour[s][cell] = scans[s]->picture.at<cv::Vec3b>(pixel / width, pixel % width);
            gridDepth[s][cell] = p.z;
        }
        for (const PlanePiece& found : FindPlanes(grid, options.planes, random))
        {
            RoomPiece piece{ {}, {}, cv::Vec3d(0, 0, 0), {} };
            cv::Point3f centre(0, 0, 0);
            for (int i : found.cells)
            {
                piece.points.push_back(roomGrid[s][i]);
                piece.colourSum += cv::Vec3d(gridColour[s][i]);
                piece.cells.emplace_back((int)s, i);
                centre += roomGrid[s][i];
            }
            centre *= 1.0f / found.cells.size();
            piece.plane = FitPlane(piece.points);
            bool merged = false;   // the same plane seen from an earlier spot? Then it's one piece.
            for (RoomPiece& other : pieces)
            {
                if (other.plane.normal.dot(piece.plane.normal) < std::cos(options.mergeDegrees * (float)CV_PI / 180) ||
                    std::abs(other.plane.DistanceTo(centre)) > options.mergeMeters) continue;
                other.points.insert(other.points.end(), piece.points.begin(), piece.points.end());
                other.cells.insert(other.cells.end(), piece.cells.begin(), piece.cells.end());
                other.colourSum += piece.colourSum;
                other.plane = FitPlane(other.points);
                merged = true;
                break;
            }
            if (!merged) pieces.push_back(piece);
        }
    }
    auto averageColour = [](const cv::Vec3d& sum, size_t count) { return cv::Vec3b(sum / (double)std::max<size_t>(1, count)); };

    // Sort the planes (09c): first by the first camera's up, then again by the room's.
    const cv::Point3f cameraUp(0, -1, 0);
    int floorPiece = -1, ceilingPiece = -1;
    std::vector<int> wallPieces;
    auto Sort = [&](const cv::Point3f& by, float tiltDegrees)
    {
        const float levelCos = std::cos(tiltDegrees * (float)CV_PI / 180), uprightSin = std::sin(tiltDegrees * (float)CV_PI / 180);
        std::vector<float> heights;
        for (const auto& grid : roomGrid)
            for (const cv::Point3f& p : grid) if (!std::isnan(p.z)) heights.push_back(p.dot(by));
        float lowest = Percentile(heights, 0.02);
        floorPiece = ceilingPiece = -1;
        wallPieces.clear();
        for (int k = 0; k < (int)pieces.size(); k++)
        {
            const Plane& plane = pieces[k].plane;
            float upness = plane.normal.dot(by);
            if (upness > levelCos)
            {
                if (floorPiece < 0 || plane.distance > pieces[floorPiece].plane.distance) floorPiece = k;
            }
            else if (-upness > levelCos)
            {
                bool highEnough = plane.distance - lowest >= options.minCeilingHeightMeters;
                if (highEnough && (ceilingPiece < 0 || plane.distance > pieces[ceilingPiece].plane.distance)) ceilingPiece = k;
            }
            else if (std::abs(upness) < uprightSin)
            {
                std::vector<float> wallHeights;
                for (const cv::Point3f& p : pieces[k].points) wallHeights.push_back(p.dot(by));
                if (Percentile(wallHeights, 0.98) - Percentile(wallHeights, 0.02) >= options.minWallHeightMeters) wallPieces.push_back(k);
            }
        }
    };
    bool anyPoints = false;
    for (const auto& grid : roomGrid) for (const cv::Point3f& p : grid) anyPoints = anyPoints || !std::isnan(p.z);
    if (!anyPoints) return room;

    // Up: from the BIGGEST level plane, floor or ceiling (13c: the farthest could be a ramp or a couch back).
    cv::Point3f up = cameraUp;
    room.upFrom = "the first camera (nothing level found, under 2 walls)";
    const float tiltCos = std::cos(options.maxTiltDegrees * (float)CV_PI / 180);
    size_t biggest = 0;
    for (const RoomPiece& piece : pieces)
    {
        float upness = piece.plane.normal.dot(cameraUp);
        if (std::abs(upness) <= tiltCos || piece.points.size() <= biggest) continue;
        biggest = piece.points.size();
        up = upness > 0 ? piece.plane.normal : -piece.plane.normal;
        room.upFrom = upness > 0 ? "the floor" : "the ceiling";
    }
    Sort(cameraUp, options.maxTiltDegrees);
    if (biggest == 0)
    {
        float bestAcross = 0.5f;
        for (size_t i = 0; i < wallPieces.size(); i++)
            for (size_t j = i + 1; j < wallPieces.size(); j++)
            {
                cv::Point3f across = pieces[wallPieces[i]].plane.normal.cross(pieces[wallPieces[j]].plane.normal);
                if (cv::norm(across) <= bestAcross) continue;
                bestAcross = (float)cv::norm(across);
                up = across * (1.0f / bestAcross);
                if (up.dot(cameraUp) < 0) up = -up;
                room.upFrom = "two walls";
            }
    }
    Sort(up, options.levelDegrees);   // now by the room's own up, tightly: a ramp is neither floor nor wall
    std::sort(wallPieces.begin(), wallPieces.end(), [&](int a, int b) { return pieces[a].points.size() > pieces[b].points.size(); });
    auto Flatten = [&](cv::Point3f v) { v -= up * up.dot(v); return v * (1.0f / (float)cv::norm(v)); };
    cv::Point3f towards = wallPieces.empty() ? Flatten(cv::Point3f(0, 0, -1)) : Flatten(pieces[wallPieces[0]].plane.normal);
    cv::Point3f right = up.cross(towards);
    room.firstCamera = cv::Matx33f(right.x, right.y, right.z, up.x, up.y, up.z, towards.x, towards.y, towards.z);
    auto ToRoom = [&](const cv::Point3f& p) { return room.firstCamera * p; };
    for (cv::Point3f& p : room.mesh.points) p = ToRoom(p);

    std::vector<float> roomHeights, roomX, roomZ;
    for (auto& grid : roomGrid)
        for (cv::Point3f& p : grid)
        {
            if (std::isnan(p.z)) continue;
            p = ToRoom(p);
            roomHeights.push_back(p.y);
            roomX.push_back(p.x);
            roomZ.push_back(p.z);
        }
    for (RoomPiece& piece : pieces) for (cv::Point3f& p : piece.points) p = ToRoom(p);
    room.floorSeen = floorPiece >= 0;
    room.ceilingSeen = ceilingPiece >= 0;
    room.floorY = room.floorSeen ? -pieces[floorPiece].plane.distance : Percentile(roomHeights, 0.02);
    room.ceilingY = room.ceilingSeen ? pieces[ceilingPiece].plane.distance : Percentile(roomHeights, 0.98);
    std::vector<std::vector<char>> taken(scans.size(), std::vector<char>(cells, 0));   // cells used by floor, ceiling, walls
    for (int k : { floorPiece, ceilingPiece })
        if (k >= 0) for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;

    // Walls (09c): each as a line on the floor plan, joined at corners.
    struct Line { cv::Point2f towards, ends[2]; float offset; cv::Vec3b colour; };
    std::vector<Line> lines;
    for (int k : wallPieces)
    {
        cv::Point3f n = ToRoom(Flatten(pieces[k].plane.normal));
        Line line;
        line.towards = cv::Point2f(n.x, n.z) / std::hypot(n.x, n.z);
        cv::Point2f along(-line.towards.y, line.towards.x);
        std::vector<float> offsets, alongs;
        for (const cv::Point3f& p : pieces[k].points)
        {
            cv::Point2f q(p.x, p.z);
            offsets.push_back(line.towards.dot(q));
            alongs.push_back(along.dot(q));
        }
        for (const auto& [s, i] : pieces[k].cells) taken[s][i] = 1;
        line.offset = Percentile(offsets, 0.5);
        line.ends[0] = line.towards * line.offset + along * Percentile(alongs, 0.02);
        line.ends[1] = line.towards * line.offset + along * Percentile(alongs, 0.98);
        line.colour = averageColour(pieces[k].colourSum, pieces[k].points.size());
        lines.push_back(line);
    }
    for (size_t i = 0; i < lines.size(); i++)
        for (size_t j = i + 1; j < lines.size(); j++)
        {
            cv::Point2f a = lines[i].towards, b = lines[j].towards;
            float determinant = a.x * b.y - a.y * b.x;
            if (std::abs(determinant) < 0.5f) continue;
            cv::Point2f crossing((lines[i].offset * b.y - lines[j].offset * a.y) / determinant,
                                 (a.x * lines[j].offset - b.x * lines[i].offset) / determinant);
            auto nearestEnd = [&](Line& line) -> cv::Point2f&
            { return cv::norm(line.ends[0] - crossing) < cv::norm(line.ends[1] - crossing) ? line.ends[0] : line.ends[1]; };
            cv::Point2f& endI = nearestEnd(lines[i]);
            cv::Point2f& endJ = nearestEnd(lines[j]);
            if (cv::norm(endI - crossing) < options.cornerSnapMeters && cv::norm(endJ - crossing) < options.cornerSnapMeters)
                endI = endJ = crossing;
        }
    for (const Line& line : lines)
    {
        cv::Point2f e0 = line.ends[0], e1 = line.ends[1];
        AddQuad(room.mesh, "clean_walls", { e0.x, room.floorY, e0.y }, { e1.x, room.floorY, e1.y }, { e1.x, room.ceilingY, e1.y },
                { e0.x, room.ceilingY, e0.y }, line.colour, { line.towards.x, 0, line.towards.y });
        room.walls.emplace_back(e0, e1);
    }

    // Floor and ceiling: rectangles over the walls, most of the points, and every camera.
    float minX = Percentile(roomX, 0.02), maxX = Percentile(roomX, 0.98);
    float minZ = Percentile(roomZ, 0.02), maxZ = Percentile(roomZ, 0.98);
    std::vector<cv::Point2f> outline;
    for (const auto& wall : room.walls) { outline.push_back(wall.first); outline.push_back(wall.second); }
    for (const Scan* scan : scans) { cv::Point3f c = ToRoom(scan->pose.shift); outline.emplace_back(c.x, c.z); }
    for (const cv::Point2f& e : outline)
    {
        minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
        minZ = std::min(minZ, e.y); maxZ = std::max(maxZ, e.y);
    }
    for (bool isFloor : { true, false })
    {
        float y = isFloor ? room.floorY : room.ceilingY;
        int k = isFloor ? floorPiece : ceilingPiece;
        AddQuad(room.mesh, isFloor ? "clean_floor" : "clean_ceiling", { minX, y, minZ }, { maxX, y, minZ }, { maxX, y, maxZ }, { minX, y, maxZ },
                k >= 0 ? averageColour(pieces[k].colourSum, pieces[k].points.size()) : cv::Vec3b(128, 128, 128), { 0, isFloor ? 1.0f : -1.0f, 0 });
    }

    // Furniture (09c), per scan, then boxes merged across scans.
    struct Box { cv::Point3f lo, hi; cv::Vec3d colourSum; size_t count; };
    std::vector<Box> boxes;
    const float margin = 0.05f;
    for (size_t s = 0; s < scans.size(); s++)
    {
        cv::Mat left(gridRows, gridColumns, CV_8U, cv::Scalar(0));
        for (int i = 0; i < cells; i++)
        {
            const cv::Point3f& p = roomGrid[s][i];
            if (std::isnan(p.z) || taken[s][i] || p.y < room.floorY + margin || p.y > room.ceilingY - margin) continue;
            bool onWall = false;   // on a wall's plane = wall, even where this scan saw too little to find it
            for (const Line& line : lines)
                onWall = onWall || std::abs(line.towards.dot(cv::Point2f(p.x, p.z)) - line.offset) < options.planes.thicknessMeters;
            if (!onWall) left.data[i] = 255;
        }
        for (int row = 0; row + 1 < gridRows; row++)
            for (int column = 0; column + 1 < gridColumns; column++)
            {
                int i = row * gridColumns + column;
                for (int neighbour : { i + 1, i + gridColumns })
                {
                    float z = gridDepth[s][i], zn = gridDepth[s][neighbour];
                    if (!left.data[i] || std::isnan(zn)) continue;
                    if (std::abs(z - zn) > 0.03f * std::min(z, zn)) left.data[z < zn ? i : neighbour] = 0;
                }
            }
        cv::Mat labels, stats, centroids;
        int count = cv::connectedComponentsWithStats(left, labels, stats, centroids);
        std::vector<std::vector<int>> blobCells(count);
        for (int i = 0; i < cells; i++) if (labels.ptr<int>()[i] > 0) blobCells[labels.ptr<int>()[i]].push_back(i);
        for (int label = 1; label < count; label++)
        {
            if ((int)blobCells[label].size() < options.minBlockCells) continue;
            std::vector<float> xs, ys, zs;
            Box box{ {}, {}, cv::Vec3d(0, 0, 0), blobCells[label].size() };
            for (int i : blobCells[label])
            {
                xs.push_back(roomGrid[s][i].x);
                ys.push_back(roomGrid[s][i].y);
                zs.push_back(roomGrid[s][i].z);
                box.colourSum += cv::Vec3d(gridColour[s][i]);
            }
            box.lo = cv::Point3f(Percentile(xs, 0.02), room.floorY, Percentile(zs, 0.02));
            box.hi = cv::Point3f(Percentile(xs, 0.98), Percentile(ys, 0.98), Percentile(zs, 0.98));
            if (box.hi.x - box.lo.x <= options.maxBlockMeters && box.hi.z - box.lo.z <= options.maxBlockMeters) boxes.push_back(box);
        }
    }
    for (bool joined = true; joined; )   // the same couch from two spots: join boxes that touch, while furniture-sized
    {
        joined = false;
        for (size_t i = 0; i < boxes.size() && !joined; i++)
            for (size_t j = i + 1; j < boxes.size() && !joined; j++)
            {
                const float iLo[3] = { boxes[i].lo.x, boxes[i].lo.y, boxes[i].lo.z }, iHi[3] = { boxes[i].hi.x, boxes[i].hi.y, boxes[i].hi.z };
                const float jLo[3] = { boxes[j].lo.x, boxes[j].lo.y, boxes[j].lo.z }, jHi[3] = { boxes[j].hi.x, boxes[j].hi.y, boxes[j].hi.z };
                bool touch = true;
                for (int a = 0; a < 3; a++) touch = touch && iLo[a] <= jHi[a] + 0.02f && jLo[a] <= iHi[a] + 0.02f;
                for (int a : { 0, 2 }) touch = touch && std::max(iHi[a], jHi[a]) - std::min(iLo[a], jLo[a]) <= options.maxBlockMeters;
                if (!touch) continue;
                boxes[i].lo = cv::Point3f(std::min(iLo[0], jLo[0]), std::min(iLo[1], jLo[1]), std::min(iLo[2], jLo[2]));
                boxes[i].hi = cv::Point3f(std::max(iHi[0], jHi[0]), std::max(iHi[1], jHi[1]), std::max(iHi[2], jHi[2]));
                boxes[i].colourSum += boxes[j].colourSum;
                boxes[i].count += boxes[j].count;
                boxes.erase(boxes.begin() + j);
                joined = true;
            }
    }
    for (const Box& box : boxes)
    {
        AddBox(room.mesh, "furniture_blocks", box.lo, box.hi, averageColour(box.colourSum, box.count));
        room.blocks.emplace_back(box.lo, box.hi);
    }
    return room;
}

bool SaveRoomObj(const std::string& path, Room room, const std::vector<cv::Point3f>& landmarks, const std::string& comment)
{
    for (const cv::Point3f& l : landmarks)
    {
        cv::Point3f centre = room.firstCamera * l, half(0.05f, 0.05f, 0.05f);
        AddBox(room.mesh, "far_landmarks", centre - half, centre + half, cv::Vec3b(255, 0, 255));
    }
    if (!SaveObj(path, room.mesh, comment + "\n# spotter format 2 (left and right as in the room, not mirrored)")) return false;
    const cv::Matx33f& t = room.firstCamera;   // how scan 1's camera sat in the room, for presentation Demo 4
    std::ofstream file(path, std::ios::app);
    file << cv::format("# first camera turn: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", t(0, 0), t(0, 1), t(0, 2), t(1, 0),
                       t(1, 1), t(1, 2), t(2, 0), t(2, 1), t(2, 2));
    return (bool)file;
}

void RoomPoints(const std::vector<const Scan*>& scans, const cv::Matx33f& turn, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours)
{
    points.clear();
    colours.clear();
    for (const Scan* scan : scans)
        for (size_t i = 0; i < scan->cloud.size(); i++)
        {
            const cv::Point3f& p = scan->cloud[i];
            if (std::isnan(p.z)) continue;
            points.push_back(turn * scan->pose(p));
            colours.push_back(scan->picture.at<cv::Vec3b>((int)i / scan->picture.cols, (int)i % scan->picture.cols));
        }
}

}
