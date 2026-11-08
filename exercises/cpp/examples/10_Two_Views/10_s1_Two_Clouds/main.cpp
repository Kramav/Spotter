// =====================================================================
//  10 · step 1 of 2 · Two clouds at once
//
//  SPACE keeps a copy of the live cloud: view A, orange. The live cloud
//  keeps drawing in blue on top. Hold the Kinect still: blue covers orange.
//  Slide it 20 cm sideways: blue slides too, but the room didn't move.
//  Each cloud is in ITS OWN camera's coordinates (metres from that camera),
//  so the same wall lands in two places.
//
//  New in this step: keeping a captured cloud; two clouds in one viewer.
//  Keys:  SPACE  capture A (again: let go of it)
//         mouse drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit
//  Next:  10_s2_Move_B (move B's points into A's coordinates)
// =====================================================================

#include <opencv2/highgui.hpp>

#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "SPACE  capture A (again: let go of it)   mouse drag / W A S D  turn   wheel / Z X  zoom   R  reset   Q or Esc  quit";

const cv::Vec3b Orange(0, 140, 255), Blue(255, 160, 0);

int main()
{
    spotter::KinectSource::Options options;
    options.colour = false;   // depth only
    spotter::KinectSource kinect(options);
    if (!kinect.IsOpen()) return 1;

    const std::string window = "10 step 1: orange = A (SPACE), blue = live";
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, 2.5f), 2.5f);
    view.Attach(window);
    spotter::PointRenderer renderer;
    std::vector<cv::Point3f> a;   // empty until SPACE
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        std::vector<cv::Point3f> live = frame.depthCamera.Cloud(frame.depthMm);
        renderer.Begin(view);
        renderer.Draw(a, Orange);
        renderer.Draw(live, Blue);
        cv::imshow(window, spotter::WithKeys(renderer.Image(), Keys));
        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') a = a.empty() ? live : std::vector<cv::Point3f>();
        view.HandleKey(key);
    }
    return 0;
}

// Try: capture A, turn the Kinect 10 degrees. Far walls jump much further than near ones:
// a turn moves things in proportion to their distance.
