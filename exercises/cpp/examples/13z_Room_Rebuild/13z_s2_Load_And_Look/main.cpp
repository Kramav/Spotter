// =====================================================================
//  13z · step 2 of 2 · One room's scans in the viewer      (no Kinect needed)
//
//  Step 1 checked that each scan loads. Now load them all (the room given
//  as the argument, or the first under room_scans) and put every scan's
//  points where its pose says, in 08's viewer, with a circle and 4 arrows
//  where each camera stood. Excluded scans are grey.
//
//  New in this step: a whole room from its folder, every scan moved by its
//  pose (pose(point) = turn * point + shift).
//  Run:   13z_s2_Load_And_Look [room folder]
//  Keys:  mouse drag / W A S D turn   wheel / Z X zoom   R reset   Q quit
//  Next:  13z_Room_Rebuild (choose rooms and scans, exclude, build .obj and .ply)
// =====================================================================

#include <cstdio>
#include <opencv2/highgui.hpp>

#include "spotter/room_folder.hpp"
#include "spotter/view.hpp"

// The keys, shown in a strip below each window (spotter::WithKeys).
const std::string Keys = "mouse drag / W A S D turn   wheel / Z X zoom   R reset   Q quit";

//std::vector<spotter::Scan> LoadAll(const std::filesystem::path& room, const spotter::RoomInfo& info, std::vector<cv::Point3f>& points, std::vector<cv::Vec3b>& colours){
//
//@param1 - room: the room folder
//@param2 - info: its room.txt
//@param3 - points: filled with every 2nd pixel of each scan, moved into the room
//@param4 - colours: filled with their colours (excluded scans grey)
//
//@brief - every scan that loads (a scan that doesn't is skipped with the reason), and every 2nd pixel of each
//         moved into the room by its pose (new). Prints the scan and point counts. O(scans x pixels).
std::vector<spotter::Scan> LoadAll(const std::filesystem::path& room, const spotter::RoomInfo& info, std::vector<cv::Point3f>& points,
                                   std::vector<cv::Vec3b>& colours)
{
    std::vector<spotter::Scan> scans;
    for (const std::filesystem::path& folder : spotter::ScanFolders(room))
    {
        spotter::Scan scan;
        int number;
        std::string problem;
        if (!spotter::LoadScan(folder, info, scan, number, problem)) { std::printf("scan %d skipped: %s\n", number, problem.c_str()); continue; }
        for (size_t i = 0; i < scan.cloud.size(); i += 2)
            if (!std::isnan(scan.cloud[i].z))
            {
                points.push_back(scan.pose(scan.cloud[i]));   // new: into the room
                colours.push_back(scan.excluded ? cv::Vec3b(90, 90, 90) : scan.picture.at<cv::Vec3b>((int)i / scan.picture.cols, (int)i % scan.picture.cols));
            }
        scans.push_back(std::move(scan));
    }
    std::printf("%s: %zu scans, %zu points shown\n", room.string().c_str(), scans.size(), points.size());
    return scans;
}

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: 13z · step 2 of 2 · One room's scans in the viewer (no Kinect needed). Step 1 checked that
//         each scan loads. Now load them all (the room given as the argument, or the first under room_scans) and
//         put every scan's points where its pose says, in 08's viewer, with a circle and 4 arrows where each
//         camera stood. Setup, main loop and clean-up are laid out in the file's sections. Returns 0 on a normal
//         exit, 1 if it couldn't start.
int main(int argc, char** argv)
{
    std::vector<spotter::SavedRoom> rooms = spotter::FindRooms(std::filesystem::absolute(argc > 1 ? argv[1] : "room_scans"));
    if (rooms.empty()) { std::printf("No saved room there (13e saves them).\n"); return 1; }
    spotter::RoomInfo info;
    std::string problem;
    if (!spotter::ReadRoomInfo(rooms[0].folder, info, problem)) { std::printf("%s\n", problem.c_str()); return 1; }
    std::vector<cv::Point3f> points;
    std::vector<cv::Vec3b> colours;
    std::vector<spotter::Scan> scans = LoadAll(rooms[0].folder, info, points, colours);

    const std::string window = "13z step 2: " + rooms[0].folder.filename().string();
    cv::namedWindow(window);
    spotter::OrbitView view(spotter::OrbitView::Options(), cv::Point3f(0, 0, 2.5f), 2.5f);
    view.Attach(window);
    spotter::PointRenderer renderer;
    while (true)
    {
        cv::Mat& image = renderer.Begin(view);
        renderer.Draw(points, colours, 2, points.size() / 600000 + 1);
        for (size_t s = 0; s < scans.size(); s++) spotter::DrawCamera(image, view, scans[s].pose, info.depthCamera, cv::Scalar(0, 255, 255), 1, std::to_string(s + 1));
        cv::imshow(window, spotter::WithKeys(image, Keys));
        int key = cv::waitKey(15);
        if (spotter::IsQuitKey(key) || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) break;
        view.HandleKey(key);
    }
    return 0;
}

// Try: a legacy room (format 1, from the old 13e) next to a new one of the same room: the old one
// is now the right way round too.
