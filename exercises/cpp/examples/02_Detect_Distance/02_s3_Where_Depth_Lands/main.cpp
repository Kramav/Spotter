// =====================================================================
//  02 · step 3 of 4 · Where each depth pixel lands in colour
//
//  The two cameras sit a few cm apart and see differently (512 x 424 vs
//  1920 x 1080, different lenses), so depth pixel (row, column) is NOT
//  colour pixel (row, column). KinectSource works out where each depth
//  pixel lands: frame.colourIndex[row * 512 + column] = colourRow * 1920
//  + colourColumn, or -1 if it lands nowhere (no reading, or outside).
//
//  This step draws every 8th depth pixel as a dot on the colour picture,
//  coloured by distance (near = red, far = blue). The dots cover less than
//  the whole picture: the depth camera sees a narrower, taller view.
//
//  New in this step: colourIndex (the "registration").
//  Keys:  Q or Esc  quit
//  Next:  02_s4_Box_Distance (use it to measure a box)
// =====================================================================

#include <algorithm>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "Q or Esc  quit";

const int DotSpacing = 8;   // draw every 8th depth pixel each way, so the dots don't cover everything

void DrawDepthDots(const spotter::RgbdFrame& frame, cv::Mat& picture)
{
    for (int row = 0; row < frame.depthMm.rows; row += DotSpacing)
        for (int column = 0; column < frame.depthMm.cols; column += DotSpacing)
        {
            int index = frame.colourIndex[row * frame.depthMm.cols + column];
            if (index < 0) continue;
            cv::Point colourPixel(index % picture.cols, index / picture.cols);
            float nearness = 1 - std::clamp((frame.depthMm.at<float>(row, column) - 500) / 5500, 0.0f, 1.0f);   // 1 at 0.5 m, 0 at 6 m
            cv::circle(picture, colourPixel, 4, cv::Scalar(255 * (1 - nearness), 0, 255 * nearness), cv::FILLED);
        }
}

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    const char* window = "02 step 3: depth pixels on the colour picture (red = near, blue = far)";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    cv::resizeWindow(window, 960, 540);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        DrawDepthDots(frame, frame.colour);
        cv::imshow(window, spotter::WithKeys(frame.colour, Keys));
        if (spotter::IsQuitKey(cv::waitKey(1))) break;
    }
    return 0;
}

// Try: wave a hand in front of a wall. The red dots stay on the hand. Look at its edge: a gap
// with no dots on the wall next to it. The depth camera couldn't see that strip of wall, because
// the hand hid it from its point of view, a few cm from the colour camera's.
