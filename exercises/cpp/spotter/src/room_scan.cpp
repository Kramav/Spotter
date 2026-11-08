#include "spotter/room_scan.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <opencv2/imgproc.hpp>

namespace spotter
{

long long CubeKey(const cv::Point3f& p, float size)
{
    auto cell = [&](float v) { return (long long)std::floor(v / size) & 0x1FFFFF; };
    return (cell(p.x) << 42) | (cell(p.y) << 21) | cell(p.z);
}

static cv::Point3f CloudAt(const Scan& scan, const cv::Point2f& pixel)
{
    const int width = scan.picture.cols, height = scan.picture.rows;
    int column = std::clamp((int)std::lround(pixel.x), 0, width - 1), row = std::clamp((int)std::lround(pixel.y), 0, height - 1);
    return scan.cloud[row * width + column];
}

Scan MakeScan(const std::vector<cv::Point3f>& cloud, const cv::Mat& picture, const ScanOptions& options)
{
    Scan scan;
    scan.cloud = cloud;
    scan.picture = picture.clone();
    const int width = picture.cols, height = picture.rows;
    scan.surface = SampleSurface(cloud, width, options.icpStep);
    cv::Mat usable(height, width, CV_8U);   // features only where there's depth and colour, clear of holes' edges
    for (int i = 0; i < width * height; i++)
        usable.data[i] = !std::isnan(cloud[i].z) && picture.at<cv::Vec3b>(i / width, i % width) != cv::Vec3b(0, 0, 0) ? 255 : 0;
    cv::erode(usable, usable, cv::Mat(), cv::Point(-1, -1), 3);
    cv::Mat grey;
    cv::cvtColor(picture, grey, cv::COLOR_BGR2GRAY);
    scan.near = FindFeatures(grey, usable, options.features);
    return scan;
}

ColourModel FitColourModel(const std::vector<cv::Point3f>& cloud, const std::vector<int>& colourIndex, int colourWidth)
{
    std::vector<double> across, down, columns, rows;
    for (size_t i = 0; i < cloud.size() && i < colourIndex.size(); i += 7)
    {
        const cv::Point3f& p = cloud[i];
        if (std::isnan(p.z) || p.z <= 0 || colourIndex[i] < 0) continue;
        across.insert(across.end(), { p.x / p.z, 1 / p.z, 1 });
        down.insert(down.end(), { p.y / p.z, 1 / p.z, 1 });
        columns.push_back(colourIndex[i] % colourWidth);
        rows.push_back(colourIndex[i] / colourWidth);
    }
    ColourModel model;
    if (columns.size() < 200) return model;
    cv::Mat x, y;
    cv::solve(cv::Mat((int)columns.size(), 3, CV_64F, across.data()), cv::Mat(columns), x, cv::DECOMP_SVD);
    cv::solve(cv::Mat((int)rows.size(), 3, CV_64F, down.data()), cv::Mat(rows), y, cv::DECOMP_SVD);
    model = { x.at<double>(0), x.at<double>(2), y.at<double>(0), y.at<double>(2), x.at<double>(1), y.at<double>(1), true };
    return model;
}

void AddFarFeatures(Scan& scan, const cv::Mat& colourGrey, const cv::Mat& noDepth, const ColourModel& model,
                    const PinholeCamera& depthCamera, const ScanOptions& options)
{
    if (!model.ok || colourGrey.empty() || noDepth.empty()) return;
    cv::Mat mask;
    cv::erode(noDepth, mask, cv::Mat(), cv::Point(-1, -1), 8);   // keep clear of where depth starts
    const PinholeCamera& lens = depthCamera;
    int left = std::max(0, (int)(model.b + model.a * (0 - lens.cx) / lens.fx));
    int right = std::min(colourGrey.cols, (int)(model.b + model.a * (lens.width - lens.cx) / lens.fx));
    int top = std::max(0, (int)(model.d + model.c * (0 - lens.cy) / lens.fy));
    int bottom = std::min(colourGrey.rows, (int)(model.d + model.c * (lens.height - lens.cy) / lens.fy));
    cv::Mat inView = cv::Mat::zeros(mask.size(), CV_8U);
    if (right > left && bottom > top) inView(cv::Rect(left, top, right - left, bottom - top)).setTo(255);
    cv::bitwise_and(mask, inView, mask);
    scan.far = FindFeatures(colourGrey, mask, options.farFeatures);
    scan.farOrigin = cv::Point3f((float)(-model.e / model.a), (float)(-model.f / model.c), 0);
    scan.farRays.clear();
    for (const cv::KeyPoint& k : scan.far.keypoints)
    {
        cv::Point3f d((float)((k.pt.x - model.b) / model.a), (float)((k.pt.y - model.d) / model.c), 1);
        scan.farRays.push_back(d * (1.0f / (float)cv::norm(d)));
    }
}

static cv::Mat AllDescriptors(const Scan& scan)
{
    cv::Mat all = scan.near.descriptors.clone();
    if (!scan.far.descriptors.empty()) all.push_back(scan.far.descriptors);
    return all;
}

// The turn that best maps directions `from` onto `to`: Kabsch without the centring.
static cv::Matx33f TurnBetween(const std::vector<cv::Point3f>& from, const std::vector<cv::Point3f>& to)
{
    cv::Matx33d together = cv::Matx33d::zeros();
    for (size_t i = 0; i < from.size(); i++)
    {
        const cv::Point3d a(from[i]), b(to[i]);
        together += cv::Matx33d(a.x * b.x, a.x * b.y, a.x * b.z, a.y * b.x, a.y * b.y, a.y * b.z, a.z * b.x, a.z * b.y, a.z * b.z);
    }
    cv::Mat s, u, vt;
    cv::SVD::compute(cv::Mat(together), s, u, vt);
    cv::Mat turn = vt.t() * u.t();
    if (cv::determinant(turn) < 0)
    {
        vt.row(2) *= -1;
        turn = vt.t() * u.t();
    }
    return cv::Matx33f(cv::Matx33d(turn));
}

// Where two rays (unit directions) pass closest: how far along each. False if they're parallel.
static bool Triangulate(const cv::Point3f& oa, const cv::Point3f& da, const cv::Point3f& ob, const cv::Point3f& db, float& sa, float& sb)
{
    cv::Point3f w = oa - ob;
    float b = da.dot(db), d = da.dot(w), e = db.dot(w), denom = 1 - b * b;
    if (denom < 1e-7f) return false;
    sa = (b * e - d) / denom;
    sb = (e - b * d) / denom;
    return true;
}

// Does a far match agree with pose h (b's camera -> a's)? The rays must meet, far away. From
// (almost) the same spot they can't be triangulated, so then only their directions are compared.
static bool FarAgrees(const Pose& h, const cv::Point3f& ob, const cv::Point3f& rb, const cv::Point3f& oa, const cv::Point3f& ra,
                      const ScanOptions& options)
{
    const float agree = options.farAgreeDegrees * (float)CV_PI / 180;
    cv::Point3f o = h(ob), d = h.turn * rb;
    float sa, sb;
    if (cv::norm(o - oa) < 0.2f || !Triangulate(oa, ra, o, d, sa, sb)) return d.dot(ra) > std::cos(agree);
    if (sa < options.farMinMeters || sb < options.farMinMeters) return false;
    return cv::norm((oa + ra * sa) - (o + d * sb)) < agree * sa;
}

FeatureMatch FeaturePose(const Scan& b, const Scan& a, std::mt19937& random, int tries, const std::vector<Landmark>& landmarks,
                         const ScanOptions& options)
{
    FeatureMatch result;
    std::vector<cv::Point3f> from, to;
    std::vector<int> fromKeypoint;
    for (auto [i, j] : MatchFeatures(b.near.descriptors, a.near.descriptors, options.matchRatio))
    {
        cv::Point3f p = CloudAt(b, b.near.keypoints[i].pt), q = CloudAt(a, a.near.keypoints[j].pt);
        if (std::isnan(p.z) || std::isnan(q.z)) continue;
        from.push_back(p);
        to.push_back(q);
        fromKeypoint.push_back(i);
    }
    std::vector<std::pair<int, int>> far = MatchFeatures(b.far.descriptors, a.far.descriptors, options.matchRatio);
    std::vector<std::pair<int, int>> marks;
    std::vector<cv::Point3f> markInA;   // the matched landmarks, in a's camera coordinates
    if (!landmarks.empty())
    {
        cv::Mat landmarkDescriptors;
        for (const Landmark& l : landmarks) landmarkDescriptors.push_back(l.descriptor);
        marks = MatchFeatures(b.far.descriptors, landmarkDescriptors, options.matchRatio);
        const cv::Matx33f back = a.pose.turn.t();
        for (const auto& m : marks) markInA.push_back(back * (landmarks[m.second].point - a.pose.shift));
    }
    const float agreeCos = std::cos(options.farAgreeDegrees * (float)CV_PI / 180);
    auto Count = [&](const Pose& h, std::vector<size_t>* nearIn, std::vector<size_t>* farIn, std::vector<size_t>* markIn)
    {
        int n = 0;
        for (size_t k = 0; k < from.size(); k++)
            if (cv::norm(h(from[k]) - to[k]) < options.featureInlierMeters) { n++; if (nearIn) nearIn->push_back(k); }
        for (size_t k = 0; k < far.size(); k++)
            if (FarAgrees(h, b.farOrigin, b.farRays[far[k].first], a.farOrigin, a.farRays[far[k].second], options)) { n++; if (farIn) farIn->push_back(k); }
        for (size_t k = 0; k < marks.size(); k++)
        {
            cv::Point3f o = h(b.farOrigin), d = h.turn * b.farRays[marks[k].first], v = markInA[k] - o;
            float length = (float)cv::norm(v);
            if (length > 1 && d.dot(v) > length * agreeCos) { n++; if (markIn) markIn->push_back(k); }
        }
        return n;
    };
    const bool nearMode = from.size() >= 3;
    if (!nearMode && far.size() < 2) return result;
    std::uniform_int_distribution<size_t> pickNear(0, std::max<size_t>(from.size(), 1) - 1), pickFar(0, std::max<size_t>(far.size(), 1) - 1);
    Pose best;
    int bestCount = 0;
    for (int t = 0; t < tries; t++)
    {
        Pose guess;
        if (nearMode)   // a pose from 3 near pairs
        {
            size_t i = pickNear(random), j = pickNear(random), k = pickNear(random);
            if (cv::norm((from[j] - from[i]).cross(from[k] - from[i])) < 0.01) continue;
            guess = Kabsch({ from[i], from[j], from[k] }, { to[i], to[j], to[k] });
        }
        else            // a turn from 2 far rays, the colour cameras assumed at the same spot
        {
            size_t i = pickFar(random), j = pickFar(random);
            const cv::Point3f &bi = b.farRays[far[i].first], &bj = b.farRays[far[j].first];
            if (cv::norm(bi.cross(bj)) < 0.05) continue;
            guess.turn = TurnBetween({ bi, bj }, { a.farRays[far[i].second], a.farRays[far[j].second] });
            guess.shift = a.farOrigin - guess.turn * b.farOrigin;
        }
        int n = Count(guess, nullptr, nullptr, nullptr);
        if (n > bestCount) { bestCount = n; best = guess; }
    }
    if (bestCount < 3) return result;
    std::vector<size_t> nearIn, farIn, markIn;
    Count(best, &nearIn, &farIn, &markIn);
    if (nearIn.size() >= 3)
    {
        std::vector<cv::Point3f> fromAgreeing, toAgreeing;
        for (size_t k : nearIn) { fromAgreeing.push_back(from[k]); toAgreeing.push_back(to[k]); }
        result.pose = Kabsch(fromAgreeing, toAgreeing);
    }
    else
    {
        std::vector<cv::Point3f> fromRays, toRays;
        for (size_t k : farIn) { fromRays.push_back(b.farRays[far[k].first]); toRays.push_back(a.farRays[far[k].second]); }
        result.pose = best;
        if (fromRays.size() >= 2)
        {
            result.pose.turn = TurnBetween(fromRays, toRays);
            result.pose.shift = a.farOrigin - result.pose.turn * b.farOrigin;
        }
    }
    result.count = (int)(nearIn.size() + farIn.size() + markIn.size());
    result.near = (int)nearIn.size();
    for (size_t k : nearIn) result.nearKeypoints.push_back(fromKeypoint[k]);
    for (size_t k : farIn) { result.farKeypoints.push_back(far[k].first); result.farPairs.push_back(far[k]); }
    for (size_t k : markIn) { result.farKeypoints.push_back(marks[k].first); result.landmarkPairs.push_back(marks[k]); }
    return result;
}

std::vector<size_t> Shortlist(const cv::Mat& descriptors, const std::vector<Scan>& scans, int k, int skip, const ScanOptions& options)
{
    cv::Mat all;
    std::vector<size_t> owner;
    for (size_t s = 0; s < scans.size(); s++)
    {
        if (scans[s].excluded || (int)s == skip) continue;
        cv::Mat both = AllDescriptors(scans[s]);
        if (both.empty()) continue;
        all.push_back(both);
        owner.insert(owner.end(), both.rows, s);
    }
    std::vector<int> votes(scans.size(), 0);
    for (const cv::DMatch& m : NearestFeatures(descriptors, all))
        if (m.distance <= options.shortlistHamming) votes[owner[m.trainIdx]]++;
    std::vector<size_t> order(scans.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return votes[a] > votes[b]; });
    while (!order.empty() && (order.size() > (size_t)k || votes[order.back()] == 0)) order.pop_back();
    return order;
}

Placement PlaceScan(const Scan& scan, const std::vector<Scan>& earlier, const PinholeCamera& depthCamera, std::mt19937& random,
                    int skip, const std::vector<Landmark>& allLandmarks, const ScanOptions& options)
{
    Placement result;
    std::vector<Landmark> landmarks;   // bug 14: placing a scan again, its own landmarks would confirm its old place
    for (const Landmark& l : allLandmarks)
        if (skip < 0 || (l.scans[0] != skip && l.scans[1] != skip)) landmarks.push_back(l);
    FeatureMatch best;
    size_t bestScan = 0;
    for (size_t s : Shortlist(AllDescriptors(scan), earlier, options.placeCandidates, skip, options))
    {
        FeatureMatch m = FeaturePose(scan, earlier[s], random, options.ransacTries, landmarks, options);
        if (m.count > best.count) { best = m; bestScan = s; }
    }
    if (best.count < options.minFeatureMatches)
    {
        result.report = cv::format("only %d feature matches agree with any earlier scan (need %d; near and far count)", best.count,
                                   options.minFeatureMatches);
        return result;
    }
    Pose rough = best.pose.Then(earlier[bestScan].pose);

    // 13d: placed by far rays (under 3 near matches), rays to landmarks pin the position too: the
    // camera sits where the landmarks' lines of sight pass closest, and the turn points the rays
    // at them from there. A few rounds, only when the landmarks are spread across the view.
    int bearings = 0;
    if (best.near < 3 && best.landmarkPairs.size() >= 3)
    {
        Pose solved = rough;
        bool spread = true;
        for (int round = 0; round < 5 && spread; round++)
        {
            cv::Matx33d A = cv::Matx33d::zeros();
            cv::Vec3d B(0, 0, 0);
            const cv::Point3f offset = solved.turn * scan.farOrigin;
            for (const auto& [i, l] : best.landmarkPairs)
            {
                cv::Point3f u = solved.turn * scan.farRays[i], L = landmarks[l].point - offset;
                cv::Matx33d across = cv::Matx33d::eye() - cv::Matx33d(u.x * u.x, u.x * u.y, u.x * u.z, u.y * u.x, u.y * u.y, u.y * u.z,
                                                                       u.z * u.x, u.z * u.y, u.z * u.z);
                A += across;
                B += across * cv::Vec3d(L.x, L.y, L.z);
            }
            cv::Mat values;
            cv::eigen(cv::Mat(A), values);
            spread = values.at<double>(2) > options.bearingMinSpread * best.landmarkPairs.size();
            if (!spread) break;
            cv::Vec3d p = A.inv() * B;
            solved.shift = cv::Point3f((float)p[0], (float)p[1], (float)p[2]);
            std::vector<cv::Point3f> rays, toLandmarks;
            for (const auto& [i, l] : best.landmarkPairs)
            {
                cv::Point3f v = landmarks[l].point - solved(scan.farOrigin);
                rays.push_back(scan.farRays[i]);
                toLandmarks.push_back(v * (1.0f / (float)cv::norm(v)));
            }
            solved.turn = TurnBetween(rays, toLandmarks);
        }
        if (spread && cv::norm(solved.shift - rough.shift) < 3) { rough = solved; bearings = (int)best.landmarkPairs.size(); }
    }
    IcpOptions icp = options.icp;
    if (best.near < 3) icp.startPairMeters = options.farIcpStartPairMeters;   // far rays' guess can be decimetres off

    // ICP's target: every earlier scan, but only what the rough pose can see of them, one point per
    // thinMeters cube. "others" = the same without the matched scan, for the second opinion.
    std::vector<cv::Point3f> target, targetNormals, others, otherNormals;
    std::unordered_set<long long> taken, takenByOthers;
    const cv::Matx33f back = rough.turn.t();
    const float marginU = options.cropMarginShare * depthCamera.width, marginV = options.cropMarginShare * depthCamera.height;
    for (size_t e = 0; e < earlier.size(); e++)
    {
        if (earlier[e].excluded || (int)e == skip) continue;
        for (size_t i = 0; i < earlier[e].surface.points.size(); i++)
        {
            cv::Point3f p = earlier[e].pose(earlier[e].surface.points[i]);
            cv::Point3f c = back * (p - rough.shift);   // as the new camera would see it
            if (c.z < 0.3f || c.z > options.maxDepthMeters + 0.5f) continue;
            std::optional<cv::Point2f> pixel = depthCamera.Project(c);
            if (!pixel || pixel->x < -marginU || pixel->x > depthCamera.width + marginU || pixel->y < -marginV ||
                pixel->y > depthCamera.height + marginV) continue;
            cv::Point3f n = earlier[e].pose.turn * earlier[e].surface.normals[i];
            if (e != bestScan && takenByOthers.insert(CubeKey(p, options.thinMeters)).second)
            {
                others.push_back(p);
                otherNormals.push_back(n);
            }
            if (!taken.insert(CubeKey(p, options.thinMeters)).second) continue;
            target.push_back(p);
            targetNormals.push_back(n);
        }
    }
    IcpResult fit = Icp(scan.surface.points, target, targetNormals, rough, icp);
    result.pose = fit.pose;
    float icpMoved = (float)cv::norm(fit.pose.shift - rough.shift);
    float icpTurned = Pose{ rough.turn.t() * fit.pose.turn, cv::Point3f() }.Degrees();
    result.report = cv::format("matched scan %zu (%d features agree: %d near, %d far, %zu landmarks); ICP: %.0f%% of points overlap, "
                               "%.1f mm apart, moved %.1f cm and %.1f deg from the features' guess",
                               bestScan + 1, best.count, best.near, (int)best.farPairs.size(), best.landmarkPairs.size(),
                               fit.pairedShare * 100, fit.rmsMeters * 1000, icpMoved * 100, icpTurned);
    bool distanceUnknown = best.near < 3 && bearings == 0 && fit.weakDirections > 0;
    if (best.near < 3)
        result.report += bearings > 0 ? cv::format("; turned by far rays, positioned by %d far landmarks", bearings)
                                      : cv::format("; turned by far rays%s", distanceUnknown ? ", but the position along the free direction is "
                                                                                              "unknown (kept from the matched scan's spot)" : "");
    if (fit.weakDirections > 0)
        result.report += cv::format(" (%d direction%s not pinned by the shape - bare walls? - the features' guess kept there)",
                                    fit.weakDirections, fit.weakDirections > 1 ? "s" : "");
    if (fit.pairedShare < options.minOverlapShare)
    {
        result.report += cv::format(": too little overlap (need %.0f%%)", options.minOverlapShare * 100);
        return result;
    }
    // Does it look right? ICP should only polish the features' guess (after far rays it may move far).
    bool icpMovedFar = best.near >= 3 && (icpMoved > options.suspiciousMoveMeters || icpTurned > options.suspiciousDegrees);
    result.suspicious = icpMovedFar || fit.rmsMeters > options.suspiciousRmsMeters || fit.pairedShare < options.suspiciousOverlap || distanceUnknown;

    // A second opinion from the rest of the room: if the matched scan was itself misplaced, this one
    // inherits its error and ICP agrees. ICP again against every other scan, with a wider gate.
    std::vector<cv::Point3f> few;
    for (size_t i = 0; i < scan.surface.points.size(); i += 4) few.push_back(scan.surface.points[i]);
    IcpOptions wide = options.icp;
    wide.startPairMeters = options.checkStartPairMeters;
    IcpResult check = others.empty() ? IcpResult{ fit.pose } : Icp(few, others, otherNormals, fit.pose, wide);
    float checkMoved = (float)cv::norm(check.pose.shift - fit.pose.shift);
    float checkTurned = Pose{ fit.pose.turn.t() * check.pose.turn, cv::Point3f() }.Degrees();
    bool roomDisagrees = check.pairedShare >= options.checkMinOverlap &&
                         (checkMoved > options.suspiciousMoveMeters || checkTurned > options.suspiciousDegrees);
    if (roomDisagrees)
    {
        result.suspicious = true;
        result.report += cv::format("; the rest of the room disagrees by %.1f cm / %.1f deg: scan %zu (its feature match) may itself be misplaced",
                                    checkMoved * 100, checkTurned, bestScan + 1);
    }
    result.stats = cv::format("%d features agree with scan %zu; ICP %.0f%% overlap, %.1f mm apart, moved %.1f cm / %.1f deg%s%s",
                              best.count, bestScan + 1, fit.pairedShare * 100, fit.rmsMeters * 1000, icpMoved * 100, icpTurned,
                              roomDisagrees ? cv::format("; the rest of the room disagrees by %.1f cm", checkMoved * 100).c_str() : "",
                              result.suspicious ? "  SUSPICIOUS" : "");
    if (result.suspicious) result.report += "; SUSPICIOUS: check it with C (colour by scan); click it and press X to exclude it";
    result.placed = true;
    result.matched = bestScan;
    result.farPairs = best.farPairs;
    return result;
}

int AddLandmarks(const Scan& b, int bIndex, const Scan& a, int aIndex, const std::vector<std::pair<int, int>>& pairs,
                 std::vector<Landmark>& landmarks, const ScanOptions& options)
{
    int added = 0;
    const float minCos = std::cos(options.minParallaxDegrees * (float)CV_PI / 180);
    for (const auto& [i, j] : pairs)
    {
        cv::Point3f ob = b.pose(b.farOrigin), db = b.pose.turn * b.farRays[i];
        cv::Point3f oa = a.pose(a.farOrigin), da = a.pose.turn * a.farRays[j];
        float sa, sb;
        if (da.dot(db) > minCos || !Triangulate(oa, da, ob, db, sa, sb)) continue;
        if (sa < options.farMinMeters || sb < options.farMinMeters || sa > options.farMaxMeters || sb > options.farMaxMeters) continue;
        cv::Point3f point = ((oa + da * sa) + (ob + db * sb)) * 0.5f;
        cv::Mat descriptor = b.far.descriptors.row(i);
        bool merged = false;
        for (Landmark& l : landmarks)
        {
            if (cv::norm(l.point - point) > options.landmarkMergeMeters ||
                cv::norm(l.descriptor, descriptor, cv::NORM_HAMMING) > options.landmarkMatchHamming) continue;
            l.point = (l.point * (float)l.seen + point) * (1.0f / (l.seen + 1));
            l.seen++;
            merged = true;
            break;
        }
        if (!merged) { landmarks.push_back({ point, descriptor.clone(), 1, { bIndex, aIndex } }); added++; }
    }
    return added;
}

LiveCheck CheckLive(const Scan& live, const std::vector<Scan>& scans, const std::vector<Landmark>& landmarks, int colourWidth,
                    const ScanOptions& options)
{
    LiveCheck result;
    std::mt19937 random(7);
    FeatureMatch best;
    for (size_t s : Shortlist(AllDescriptors(live), scans, 2, -1, options))
    {
        FeatureMatch m = FeaturePose(live, scans[s], random, options.liveRansacTries, landmarks, options);
        if (m.count > best.count) { best = m; result.scan = (int)s; }
    }
    result.anchors = best.count;
    result.farAnchors = best.count - best.near;
    std::vector<char> isAnchor(live.near.keypoints.size(), 0), isFarAnchor(live.far.keypoints.size(), 0);
    for (int k : best.nearKeypoints) isAnchor[k] = 1;
    for (int k : best.farKeypoints) isFarAnchor[k] = 1;
    const int width = live.picture.cols, height = live.picture.rows;
    for (size_t k = 0; k < live.near.keypoints.size() && !live.colourIndex.empty(); k++)
    {
        const cv::Point2f& pt = live.near.keypoints[k].pt;
        int column = std::clamp((int)std::lround(pt.x), 0, width - 1), row = std::clamp((int)std::lround(pt.y), 0, height - 1);
        int c = live.colourIndex[row * width + column];
        if (c < 0) continue;
        (isAnchor[k] ? result.anchorPixels : result.otherPixels).emplace_back((float)(c % colourWidth), (float)(c / colourWidth));
    }
    for (size_t k = 0; k < live.far.keypoints.size(); k++)   // far features are colour pixels already
        (isFarAnchor[k] ? result.farAnchorPixels : result.otherPixels).push_back(live.far.keypoints[k].pt);
    return result;
}

Recommendation RecommendNext(const std::vector<Scan>& scans, const PinholeCamera& lens, const ScanOptions& options)
{
    Recommendation best;
    std::vector<size_t> spots;
    for (size_t s = 0; s < scans.size(); s++) if (!scans[s].excluded) spots.push_back(s);
    if (spots.empty()) return best;

    // What's been seen: every included scan's points, one per coverage cube.
    std::vector<cv::Point3f> map;
    std::unordered_set<long long> taken;
    for (size_t s : spots)
        for (const cv::Point3f& p : scans[s].surface.points)
        {
            cv::Point3f q = scans[s].pose(p);
            if (taken.insert(CubeKey(q, options.coverageCellMeters)).second) map.push_back(q);
        }
    // Up: the scans' camera "up" (-Y), averaged, so the suggested turns are about the room's vertical.
    cv::Point3f up(0, 0, 0);
    for (size_t s : spots) up += scans[s].pose.turn * cv::Point3f(0, -1, 0);
    up *= 1.0f / (float)cv::norm(up);
    auto Lands = [&](const cv::Point3f& c, float& u, float& v)
    {
        std::optional<cv::Point2f> pixel = lens.Project(c);
        if (!pixel) return false;
        u = pixel->x;
        v = pixel->y;
        return u >= 0 && v >= 0 && u < lens.width && v < lens.height;
    };

    const int columns = 16, rows = 12;
    const cv::Point3f last = scans[spots.back()].pose.shift;
    float bestScore = -1e9f, bestNew = 0;
    for (size_t s : spots)
    {
        const Pose& spot = scans[s].pose;
        for (float yaw : { -90.0f, -60.0f, -30.0f, 30.0f, 60.0f, 90.0f, 180.0f })
            for (float pitch : { -20.0f, 0.0f, 20.0f })
            {
                Pose candidate;
                candidate.turn = TurnAbout(cv::Vec3d(up.x, up.y, up.z) * (yaw * CV_PI / 180)) * spot.turn * TurnAbout(cv::Vec3d(pitch * CV_PI / 180, 0, 0));
                candidate.shift = spot.shift;
                // A cell of the candidate's view is known if a seen point lands in it...
                std::vector<char> known(columns * rows, 0);
                const cv::Matx33f back = candidate.turn.t();
                for (const cv::Point3f& p : map)
                {
                    cv::Point3f c = back * (p - candidate.shift);
                    float u, v;
                    if (c.z < 0.5f || c.z > 4.5f || !Lands(c, u, v)) continue;
                    known[(int)(v * rows / lens.height) * columns + (int)(u * columns / lens.width)] = 1;
                }
                // ...or if a scan from (nearly) this spot already looked that way: a window reads
                // nothing, and would otherwise be recommended for ever.
                for (int cell = 0; cell < columns * rows; cell++)
                {
                    if (known[cell]) continue;
                    cv::Point3f ray = candidate.turn * lens.Unproject(cv::Point2f((cell % columns + 0.5f) * lens.width / columns - 0.5f,
                                                                                  (cell / columns + 0.5f) * lens.height / rows - 0.5f), 1.0f);
                    for (size_t e : spots)
                    {
                        if (cv::norm(scans[e].pose.shift - candidate.shift) > options.seenFromMeters) continue;
                        cv::Point3f d = scans[e].pose.turn.t() * ray;
                        float u, v;
                        if (d.z > 0 && Lands(d, u, v)) { known[cell] = 1; break; }
                    }
                }
                float knownShare = (float)std::count(known.begin(), known.end(), 1) / (columns * rows);
                if (knownShare < options.recommendMinOverlap) continue;
                float score = (1 - knownShare) - options.recommendDistancePenalty * (float)cv::norm(spot.shift - last);
                if (score <= bestScore) continue;
                bestScore = score;
                bestNew = 1 - knownShare;
                best.pose = candidate;
                bool right = (candidate.turn * cv::Point3f(0, 0, 1)).dot(spot.turn * cv::Point3f(1, 0, 0)) > 0;
                std::string turnText = yaw == 180 ? "turn round" : cv::format("turn %.0f deg %s", std::abs(yaw), right ? "right" : "left");
                std::string tiltText = pitch == 0 ? "" : cv::format(", tilt %.0f deg %s", std::abs(pitch), pitch > 0 ? "up" : "down");
                best.text = cv::format("Next: at spot %zu, %s%s (about %.0f%% new)", s + 1, turnText.c_str(), tiltText.c_str(), bestNew * 100);
            }
    }
    best.found = bestScore > -1e9f && bestNew >= options.recommendMinNew;
    if (!best.found) best.text = "Next: these spots have seen it all - move somewhere new, looking back at scanned parts, or press E";
    // ponytail: only existing spots are suggested (a new position could be inside furniture).
    return best;
}

}
