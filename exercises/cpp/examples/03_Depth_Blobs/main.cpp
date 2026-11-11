// =====================================================================
//  03 · Depth Blobs  (no AI)
//
//  Find "something new in the room" from depth alone, and measure its
//  distance and real size in metres. This is the guide's no-AI rung (§5.6),
//  the classic Kinect people-counter:
//
//    1. Remember the empty room's depth once (the "background").
//    2. Every pixel now noticeably CLOSER than the background is foreground.
//    3. Clean up the speckles, then group touching foreground pixels into blobs.
//    4. Each blob: median depth = distance; pixel size * depth / focal length
//       = real width and height (§5.3).
//
//  It doesn't know WHAT anything is, but it ignores lighting and shadows and
//  works in a completely dark room, unlike example 02.
//
//  Window: grey depth, foreground in green, a box and label per blob.
//  Keys:   B  re-record the background (step out of view first)
//          Q or Esc  quit
// =====================================================================

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/stats.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "B  re-record the background (step out of view first)   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// A pixel is foreground when it is at least this much closer than the background.
// Too small: sensor noise shows up as blobs. Too big: a hand flat on a table disappears.
const float ForegroundThresholdMm = 100;

// Blobs with fewer pixels than this are ignored (a person at 3 m covers a few thousand).
const int MinBlobPixels = 400;

// Size of the "opening" filter that removes isolated foreground speckles, in pixels.
// 1 = no cleanup. Try 1, then 5, then 11 and watch the green.
const int CleanupSizePixels = 5;

const double SecondsBeforeBackground = 3;   // to step out of view before the background is recorded
const float NearMm = 500, FarMm = 6000;     // the grey scale, as in example 01
const int ReportEveryMs = 1000;
const std::string Window = "03 depth blobs  (B = new background, Q quits)";

// ---------------------------------------------------------------------
//  The new idea, step 1: remember the empty room
// ---------------------------------------------------------------------

struct Background
{
    cv::Mat depthMm;   // empty until recorded
    std::chrono::steady_clock::time_point countdownStart = std::chrono::steady_clock::now();

    //double SecondsLeft(){
    //
    //@brief - the seconds left on the countdown to recording the background (negative once it's over). O(1).
    double SecondsLeft() const
    {
        return SecondsBeforeBackground - std::chrono::duration<double>(std::chrono::steady_clock::now() - countdownStart).count();
    }
    //void Restart(){
    //
    //@brief - forgets the recorded background and starts the countdown again; prints what to do (step out of view).
    //         O(1).
    void Restart()
    {
        depthMm.release();
        countdownStart = std::chrono::steady_clock::now();
        std::printf("Recording the background in %.0f s: step out of view.\n", SecondsBeforeBackground);
    }
    //void RecordWhenDue(const cv::Mat& liveMm){
    //
    //@param1 - liveMm: this frame's depth, mm
    //
    //@brief - once the countdown is over and nothing is recorded yet, keeps a copy of this frame as the empty room.
    //         Otherwise does nothing. O(pixels) on the frame it records.
    void RecordWhenDue(const cv::Mat& liveMm)
    {
        if (!depthMm.empty() || SecondsLeft() > 0) return;
        depthMm = liveMm.clone();
        std::printf("Background recorded. Walk in!\n");
    }
};

// ---------------------------------------------------------------------
//  Steps 2-4: foreground, blobs, measurements
// ---------------------------------------------------------------------

//cv::Mat Foreground(const cv::Mat& liveMm, const Background& background){
//
//@param1 - liveMm: this frame's depth, mm
//@param2 - background: the recorded empty room
//
//@brief - step 2, the guide's rule: (background - depth) > ForegroundThresholdMm AND depth has a reading. Where
//         the background had none (0) the difference is negative, so those pixels never count (experiment idea 3).
//         All zero before it is recorded. O(pixels).
cv::Mat Foreground(const cv::Mat& liveMm, const Background& background)
{
    cv::Mat mask(liveMm.size(), CV_8U, cv::Scalar(0));
    if (background.depthMm.empty()) return mask;
    mask = (background.depthMm - liveMm > ForegroundThresholdMm) & (liveMm > 0);
    return mask;
}

struct Blob
{
    cv::Rect box;   // in the depth picture
    int pixels;
    double meters, widthMeters, heightMeters;
    cv::Point3f centre;
};

//std::vector<Blob> FindBlobs(const cv::Mat& foreground, const cv::Mat& liveMm, const spotter::PinholeCamera& camera, cv::Mat& cleaned){
//
//@param1 - foreground: a 255/0 foreground mask
//@param2 - liveMm: this frame's depth, mm
//@param3 - camera: the depth camera, for real sizes
//@param4 - cleaned: set to the mask after cleaning
//
//@brief - steps 3 and 4: opens the mask to drop specks, labels touching pixels, and for each group of at least
//         MinBlobPixels measures the median depth of its own pixels, its real width and height (pixels x distance
//         / focal length) and its 3D centre. O(pixels).
std::vector<Blob> FindBlobs(const cv::Mat& foreground, const cv::Mat& liveMm, const spotter::PinholeCamera& camera, cv::Mat& cleaned)
{
    cv::Mat shape = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(CleanupSizePixels, CleanupSizePixels));
    cv::morphologyEx(foreground, cleaned, cv::MORPH_OPEN, shape);
    cv::Mat labels, stats, centroids;
    int labelCount = cv::connectedComponentsWithStats(cleaned, labels, stats, centroids, 8);

    std::vector<Blob> blobs;
    for (int label = 1; label < labelCount; label++)   // label 0 is everything that isn't foreground
    {
        Blob blob;
        blob.pixels = stats.at<int>(label, cv::CC_STAT_AREA);
        if (blob.pixels < MinBlobPixels) continue;
        blob.box = cv::Rect(stats.at<int>(label, cv::CC_STAT_LEFT), stats.at<int>(label, cv::CC_STAT_TOP),
                            stats.at<int>(label, cv::CC_STAT_WIDTH), stats.at<int>(label, cv::CC_STAT_HEIGHT));

        // The median depth of THIS blob's pixels (not the whole rectangle around it).
        std::vector<float> depthsMm;
        for (int y = blob.box.y; y < blob.box.br().y; y++)
            for (int x = blob.box.x; x < blob.box.br().x; x++)
                if (labels.at<int>(y, x) == label && liveMm.at<float>(y, x) > 0) depthsMm.push_back(liveMm.at<float>(y, x));
        if (depthsMm.empty()) continue;
        blob.meters = spotter::Median(depthsMm) / 1000;

        // Real size (§5.3): something w pixels wide at depth Z is about w * Z / fx wide.
        blob.widthMeters = blob.box.width * blob.meters / camera.fx;
        blob.heightMeters = blob.box.height * blob.meters / camera.fy;
        blob.centre = camera.Unproject(cv::Point2f((float)centroids.at<double>(label, 0), (float)centroids.at<double>(label, 1)),
                                       (float)blob.meters);
        blobs.push_back(blob);
    }
    return blobs;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

//std::string Label(const Blob& blob){
//
//@param1 - blob: a measured blob
//
//@brief - "1.23m  0.45 x 1.70m": its distance, width and height. O(1).
std::string Label(const Blob& blob) { return cv::format("%.2fm  %.2f x %.2fm", blob.meters, blob.widthMeters, blob.heightMeters); }

//void Show(const cv::Mat& liveMm, const cv::Mat& foreground, const std::vector<Blob>& blobs, const Background& background, const spotter::FpsMeter& fps){
//
//@param1 - liveMm: this frame's depth, mm
//@param2 - foreground: the foreground mask
//@param3 - blobs: the blobs found
//@param4 - background: the recorded background, for the countdown
//@param5 - fps: the frame-rate meter
//
//@brief - the depth picture in grey by distance, foreground pixels green, a yellow box and label per blob, a red
//         countdown until the background is recorded, and the fps; shown with the key strip. O(pixels).
void Show(const cv::Mat& liveMm, const cv::Mat& foreground, const std::vector<Blob>& blobs, const Background& background,
          const spotter::FpsMeter& fps)
{
    // Grey by distance; foreground pixels green (still brighter when nearer).
    cv::Mat picture = spotter::DepthPicture(liveMm, NearMm, FarMm);
    for (int i = 0; i < (int)liveMm.total(); i++)
        if (foreground.data[i]) picture.at<cv::Vec3b>(i) = cv::Vec3b(0, picture.at<cv::Vec3b>(i)[1], 0);

    const cv::Scalar yellow(0, 255, 255);
    for (const Blob& blob : blobs)
    {
        cv::rectangle(picture, blob.box, yellow, 1);
        cv::putText(picture, Label(blob), cv::Point(blob.box.x, std::max(blob.box.y - 4, 12)), cv::FONT_HERSHEY_SIMPLEX, 0.4, yellow, 1);
    }
    if (background.depthMm.empty())
        cv::putText(picture, cv::format("background in %.0f s", std::ceil(background.SecondsLeft())), cv::Point(10, 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    fps.Draw(picture);
    cv::imshow(Window, spotter::WithKeys(picture, Keys));
}

//void Report(const cv::Mat& foreground, const cv::Mat& cleaned, const std::vector<Blob>& blobs){
//
//@param1 - foreground: the raw mask
//@param2 - cleaned: the mask after cleaning
//@param3 - blobs: the blobs found
//
//@brief - console lines: foreground pixels before and after cleanup, then each blob's pixels, label and centre.
//         O(pixels + blobs).
void Report(const cv::Mat& foreground, const cv::Mat& cleaned, const std::vector<Blob>& blobs)
{
    std::printf("foreground pixels: %d raw, %d after cleanup | %zu blobs big enough\n",
                cv::countNonZero(foreground), cv::countNonZero(cleaned), blobs.size());
    for (const Blob& blob : blobs)
        std::printf("  %6d px  %-22s  centre x%+.2f y%+.2f\n", blob.pixels, Label(blob).c_str(), blob.centre.x, blob.centre.y);
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

//int main(){
//
//@brief - the program: 03 · Depth Blobs (no AI). Find "something new in the room" from depth alone, and measure
//         its distance and real size in metres. This is the guide's no-AI rung (§5.6), the classic Kinect people-
//         counter. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal exit,
//         1 if it couldn't start.
int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    spotter::KinectSource::Options kinectOptions;
    kinectOptions.colour = false;   // depth only: this works in the dark
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    cv::namedWindow(Window, cv::WINDOW_NORMAL);
    cv::resizeWindow(Window, 1024, 848);
    Background background;
    background.Restart();
    spotter::FpsMeter fps;
    spotter::Every report(ReportEveryMs);
    spotter::RgbdFrame frame;

    while (kinect.Next(frame))
    {
        background.RecordWhenDue(frame.depthMm);                         // step 1
        cv::Mat foreground = Foreground(frame.depthMm, background);      // step 2
        cv::Mat cleaned;
        std::vector<Blob> blobs = FindBlobs(foreground, frame.depthMm, frame.depthCamera, cleaned);   // steps 3 + 4

        fps.Tick();
        Show(frame.depthMm, foreground, blobs, background, fps);
        if (report.Due() && !background.depthMm.empty()) Report(foreground, cleaned, blobs);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'b') background.Restart();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Stand in view and read your height off the label. Expect roughly your real height
//     (guide §7 test 5). Now crouch, raise an arm, hold a chair: the box measures the outline.
//  2. Turn the lights off. Everything still works. Then try example 02 in the dark.
//  3. Point the Kinect at an open doorway or a window, record the background, and walk into
//     the region that was red (no reading). You don't show up there. Why? (Step 2's rule.)
//     Try treating background 0 as "very far" (8000 mm) and see what changes.
//  4. Set ForegroundThresholdMm = 20. The floor and walls start to flicker green: that's
//     the sensor's noise. Find the smallest value that stays clean at your distances.
//  5. Set CleanupSizePixels = 1 and compare the "raw" and "after cleanup" pixel counts.
//  6. Put a box on the floor, press B, then remove it. Nothing appears where it was: only
//     "closer than before" counts. What would detect objects that were taken AWAY? (07c.)
//  7. Your blob is a little smaller than you: the red outline (edge pixels with no reading)
//     is never foreground. Compare the measured width with something of known size.
