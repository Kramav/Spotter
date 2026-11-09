// =====================================================================
//  Demo 5 · How Tall Are You?         (presentation; built from 03)
//
//  Step in front of the Kinect and your height appears over your head,
//  in metres, from depth alone (no AI, works in the dark). The tallest
//  visitor so far stays in the corner: a number only, no picture.
//
//  As in 03: remember the empty view's depth, everything clearly nearer
//  is a blob (spotter::StillBackground). New: the height is the 3D
//  distance from the blob's top row to its bottom row, each with its own
//  depth, so it holds when the Kinect is tilted (03 multiplied the box's
//  pixel height by one depth).
//
//  Set-up: the Kinect at about chest height, visitors 2.5-3.5 m away so
//  head AND feet are in view (mark the spot). Start it with nobody in
//  view; B re-records the empty view. Nothing is saved.
//
//  Keys:   B  re-record the background (clear the view first)
//          T  reset "tallest today"     F  fullscreen on/off     Q or Esc  quit
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip along the bottom of the screen (spotter::WithKeys).
const std::string Keys = "B  relearn the empty view (clear it first)   T  reset tallest   F  fullscreen   Q or Esc  quit";

// ---------------------------------------------------------------------
//  Settings: change these and re-run
// ---------------------------------------------------------------------

// 03's blobs: at least this much nearer than the empty view, cleaned by an opening, big enough.
const float ForegroundThresholdMm = 100;
const int CleanupSizePixels = 5;
const int MinBlobPixels = 400;            // a person at 3 m covers a few thousand
const double SecondsBeforeBackground = 3;

// Heights outside this range are not counted for "tallest today" (two people merged, a chair).
const double MinPersonMeters = 1.0, MaxPersonMeters = 2.3;
const int MinRowReadings = 3;             // rows of the blob with fewer readings are ragged edges

const float NearMm = 500, FarMm = 4500;   // the grey scale, as in example 01
const bool StartFullscreen = true;
const std::string Window = "Demo 5 - how tall are you";

// ---------------------------------------------------------------------
//  The new idea: height from the top and bottom rows, each in 3D
// ---------------------------------------------------------------------

// One row of the blob as a 3D point: the average of its foreground pixels, each from its own depth.
std::optional<cv::Point3f> RowPoint(int row, const cv::Rect& box, const cv::Mat& mask, const cv::Mat& liveMm, const spotter::PinholeCamera& camera)
{
    cv::Point3f sum(0, 0, 0);
    int count = 0;
    for (int x = box.x; x < box.br().x; x++)
        if (mask.at<uchar>(row, x) && liveMm.at<float>(row, x) > 0)
        {
            sum += camera.Unproject(cv::Point2f((float)x, (float)row), liveMm.at<float>(row, x) / 1000);
            count++;
        }
    if (count < MinRowReadings) return std::nullopt;
    return sum * (1.0f / count);
}

struct Person
{
    cv::Rect box;
    double heightMeters = 0;
    bool wholeInView = false;   // head and feet both seen
};

Person Measure(const cv::Rect& box, const cv::Mat& mask, const cv::Mat& liveMm, const spotter::PinholeCamera& camera)
{
    Person person{ box };
    int top = box.y, bottom = box.br().y - 1;
    std::optional<cv::Point3f> head, feet;
    while (top < bottom && !(head = RowPoint(top, box, mask, liveMm, camera))) top++;
    while (bottom > top && !(feet = RowPoint(bottom, box, mask, liveMm, camera))) bottom--;
    if (!head || !feet) return person;   // no measurement (legacy measured to the camera here: review bug 12)
    person.heightMeters = cv::norm(*head - *feet);
    person.wholeInView = box.y > 2 && box.br().y < liveMm.rows - 2;
    return person;
}

// ---------------------------------------------------------------------
//  Output
// ---------------------------------------------------------------------

void Show(const cv::Mat& liveMm, const cv::Mat& mask, const std::vector<Person>& people, const spotter::StillBackground& background,
          double tallest, const spotter::FpsMeter& fps)
{
    cv::Mat picture = spotter::DepthPicture(liveMm, NearMm, FarMm);
    for (int i = 0; i < (int)liveMm.total(); i++)
        if (mask.data[i]) picture.at<cv::Vec3b>(i) = cv::Vec3b(0, picture.at<cv::Vec3b>(i)[1], 0);   // visitors green
    const cv::Scalar yellow(0, 255, 255);
    for (const Person& p : people)
    {
        cv::rectangle(picture, p.box, yellow, 1);
        cv::putText(picture, p.wholeInView ? cv::format("%.2f m", p.heightMeters) : "step back", cv::Point(p.box.x, std::max(p.box.y - 8, 24)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, yellow, 2);
    }
    if (!background.Recorded())
        cv::putText(picture, cv::format("background in %.0f s", std::ceil(background.SecondsLeft())), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX,
                    0.6, cv::Scalar(0, 0, 255), 2);
    if (tallest > 0)
        cv::putText(picture, cv::format("tallest today: %.2f m", tallest), cv::Point(picture.cols - 215, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
    cv::putText(picture, "live only: nothing is recorded", cv::Point(picture.cols - 200, picture.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                cv::Scalar(160, 160, 160), 1);
    fps.Draw(picture);
    cv::imshow(Window, spotter::WithKeys(picture, Keys));
}

// ---------------------------------------------------------------------
//  Setup and main operation
// ---------------------------------------------------------------------

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    spotter::KinectSource::Options kinectOptions;
    kinectOptions.colour = false;   // depth alone: works in the dark
    spotter::KinectSource kinect(kinectOptions);
    if (!kinect.IsOpen()) return 1;

    spotter::OpenExhibitWindow(Window, cv::Size(1024, 848), StartFullscreen);
    spotter::StillBackground background(SecondsBeforeBackground);
    spotter::FpsMeter fps;
    spotter::RgbdFrame frame;
    double tallest = 0;

    while (kinect.Next(frame))
    {
        background.RecordWhenDue(frame.depthMm);
        std::vector<cv::Rect> boxes;
        cv::Mat mask = background.Foreground(frame.depthMm, ForegroundThresholdMm, CleanupSizePixels, MinBlobPixels, boxes);
        std::vector<Person> people;
        for (const cv::Rect& box : boxes) people.push_back(Measure(box, mask, frame.depthMm, frame.depthCamera));
        for (const Person& p : people)
            if (p.wholeInView && p.heightMeters >= MinPersonMeters && p.heightMeters <= MaxPersonMeters) tallest = std::max(tallest, p.heightMeters);
        fps.Tick();
        Show(frame.depthMm, mask, people, background, tallest, fps);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == 'f') spotter::ToggleFullscreen(Window);
        if (key == 't') tallest = 0;
        if (key == 'b') background.Restart();
    }
    return 0;
}

// ---------------------------------------------------------------------
//  Experiment ideas
// ---------------------------------------------------------------------
//  1. Check it on someone who knows their height. Usually a few cm short: hair gives no depth
//     reading, and the soles of the feet are within ForegroundThresholdMm of the floor.
//  2. Tilt the Kinect down 15 deg and measure again: the 3D height barely changes. 03's box
//     method (pixel height x one depth) does.
//  3. Raise an arm: you "grow". The blob measures the outline, not the person.
//  4. Two people side by side, touching: one blob, one height, and no "tallest" (MaxPersonMeters).
//  5. Turn the lights off: it still works. Then try Demo 3 in the dark.
