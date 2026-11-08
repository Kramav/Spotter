// =====================================================================
//  07d · step 2 of 3 · Remember what the room looks like
//
//  07c's depth background (spotter::AdaptiveBackground) says which pixels
//  have something in front of the room. Everywhere else, the colour we see
//  IS the room: blend it into a colour memory. The memory window keeps the
//  bare room even while you stand in front of it. Black in the memory =
//  never seen yet.
//
//  New in this step: a colour memory that learns only where nothing is in the way.
//  Keys:  SPACE  forget and start over     Q or Esc  quit
//  Next:  07d_s3_Paint_Out (use the memory to paint a person out)
// =====================================================================

#include <opencv2/highgui.hpp>

#include "spotter/depth.hpp"
#include "spotter/source_freenect2.hpp"
#include "spotter/view.hpp"

const float BlendRate = 0.1f;   // per frame, how fast the memory follows the room

void Learn(cv::Mat& memory, const cv::Mat& registered, const cv::Mat& foreground)
{
    for (int i = 0; i < (int)registered.total(); i++)
    {
        cv::Vec3b live = registered.at<cv::Vec3b>(i);
        if (foreground.data[i] || live == cv::Vec3b(0, 0, 0)) continue;   // something in the way, or the colour camera can't see it
        cv::Vec3f& remembered = memory.at<cv::Vec3f>(i);
        if (remembered == cv::Vec3f(0, 0, 0)) remembered = live;          // first time
        else remembered += BlendRate * (cv::Vec3f(live) - remembered);
    }
}

int main()
{
    spotter::KinectSource kinect(spotter::KinectSource::Options{});
    if (!kinect.IsOpen()) return 1;

    spotter::AdaptiveBackground background(spotter::AdaptiveBackground::Options{});
    cv::Mat memory = cv::Mat::zeros(424, 512, CV_32FC3);
    spotter::RgbdFrame frame;
    while (kinect.Next(frame))
    {
        cv::Mat foreground = background.Update(frame.depthMm);   // 255 = in front of the remembered room
        Learn(memory, frame.registered, foreground);
        cv::Mat shown;
        memory.convertTo(shown, CV_8UC3);
        cv::imshow("07d step 2: live", frame.registered);
        cv::imshow("07d step 2: colour memory (the room without you)", shown);

        int key = cv::waitKey(1);
        if (spotter::IsQuitKey(key)) break;
        if (key == ' ') { background.Forget(); memory.setTo(0); }
    }
    return 0;
}

// Try: start with the room empty, then walk in: the memory window keeps the room. Now press SPACE
// while you're in view: where you stand stays black in the memory until you step away.
