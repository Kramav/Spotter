// =====================================================================
//  13z · step 1 of 2 · Find the saved rooms                (no Kinect needed)
//
//  A saved room is a folder holding a room.txt (13e writes one). This
//  looks under a folder (the first argument, or the repo's room_scans/)
//  up to 3 folders deep, lists each room, then checks every scan of each:
//  does it load (spotter::LoadScan), and if not, why not. A scan cut off
//  by a crash has no pose.txt, or a cloud.bin that doesn't match it.
//
//  New in this step: FindRooms, ReadRoomInfo, ScanFolders, LoadScan.
//  Run:   13z_s1_List_Rooms [folder]
//  Next:  13z_s2_Load_And_Look (one room's scans in the 3D viewer)
// =====================================================================

#include <cstdio>

#include "spotter/room_folder.hpp"

//int main(int argc, char** argv){
//
//@param1 - argc: how many command-line words there are
//@param2 - argv: the words; the file's header says what it accepts
//
//@brief - the program: 13z · step 1 of 2 · Find the saved rooms (no Kinect needed). A saved room is a folder
//         holding a room.txt (13e writes one). Setup, main loop and clean-up are laid out in the file's sections.
//         Returns 0 on a normal exit.
int main(int argc, char** argv)
{
    const std::filesystem::path where = std::filesystem::absolute(argc > 1 ? argv[1] : SPOTTER_REPO_ROOT "/room_scans");
    std::vector<spotter::SavedRoom> rooms = spotter::FindRooms(where);
    std::printf("%zu saved rooms in %s\n", rooms.size(), where.string().c_str());
    for (const spotter::SavedRoom& room : rooms)
    {
        std::printf("\n%s  (%s, saved %s, format %d, %.0f MB)\n", room.folder.filename().string().c_str(), room.program.c_str(),
                    room.saved.c_str(), room.format, room.megabytes);
        spotter::RoomInfo info;
        std::string problem;
        if (!spotter::ReadRoomInfo(room.folder, info, problem)) { std::printf("  can't be read: %s\n", problem.c_str()); continue; }
        std::printf("  depth camera fx %.1f, cx %.1f%s; %s\n", info.depthCamera.fx, info.depthCamera.cx,
                    info.format == 1 ? " (unmirrored: format 1 was mirrored)" : "", info.finished ? "finished" : "not finished (no up-to-date points.ply)");
        for (const std::filesystem::path& folder : spotter::ScanFolders(room.folder))
        {
            spotter::Scan scan;
            int number;
            if (spotter::LoadScan(folder, info, scan, number, problem))
                std::printf("  scan %3d  camera at (%+.2f, %+.2f, %+.2f) m%s%s\n", number, scan.pose.shift.x, scan.pose.shift.y, scan.pose.shift.z,
                            scan.excluded ? "  excluded" : "", scan.suspicious ? "  suspicious" : "");
            else std::printf("  scan %3d  skipped: %s\n", number, problem.c_str());
        }
    }
    std::printf("\nPress Enter to close.\n");
    std::getchar();
    return 0;
}

// Try: copy a room folder and delete one scan's pose.txt in the copy: that scan is skipped,
// the rest still load.
