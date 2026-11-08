// =====================================================================
//  10 · step 2 of 2 · Move B into A's coordinates
//
//  Step 1's blue cloud is in the moved camera's coordinates. Tell the
//  program how the camera moved (the settings) and turn and shift every
//  blue point by that, before drawing it:
//      point in A = turn(point in B) + where B's camera stands
//  Right pose: blue lands on orange, one room. Wrong: walls show double.
//
//  New in this step: a pose (shift + turn left) and moving points by it.
//  Keys:  SPACE  capture A (again: let go of it)
//         mouse drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit
//  Next:  10_Two_Views (keys to nudge the pose, and a number for how well they meet)
// =====================================================================

#include <cmath>
#include <opencv2/highgui.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE  capture A (again: let go of it)   mouse drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit";

// How camera B moved from A, in A's directions. Turn = how far B is turned to the left.
const float MoveRightMeters = 0.20f, MoveUpMeters = 0, MoveForwardMeters = 0, TurnLeftDegrees = 0;
const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);

// New: a point B sees -> where it is for A.
cv::Point3f ToA(const cv::Point3f& p)
{
    const float turn = TurnLeftDegrees * (float)CV_PI / 180;
    float x = std::cos(turn) * p.x - std::sin(turn) * p.z;   // 1. turn about the vertical axis
    float z = std::sin(turn) * p.x + std::cos(turn) * p.z;
    return cv::Point3f(x + MoveRightMeters, p.y - MoveUpMeters, z + MoveForwardMeters);   // 2. shift (Y points down)
}

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    const std::string window = "10 step 2: orange = A (SPACE), blue = live moved by the pose";
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, 2.5f), 2.5f);
    view.Attach(window);
    spotter::PointRenderer renderer;
    std::vector<cv::Point3f> a;
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<cv::Point3f> live = frame.depthCamera.Cloud(frame.depthMm);
        if (!a.empty()) for (cv::Point3f& p : live) p = ToA(p);   // NaN stays NaN
        renderer.Begin(view);
        renderer.Draw(a, Orange);
        renderer.Draw(live, Blue);
        cv::imshow(window, spotter::WithKeys(renderer.Image(), Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') a = a.empty() ? frame.depthCamera.Cloud(frame.depthMm) : std::vector<cv::Point3f>();
        view.HandleKey(key);
    }
    return 0;
}

// Try: capture A, slide the Kinect 20 cm right (keep it pointing the same way). Blue should sit on
// orange. Now set MoveRightMeters = 0.10 and re-run: every wall doubles 10 cm apart.
