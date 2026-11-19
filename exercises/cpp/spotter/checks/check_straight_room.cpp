// Self-check for Demo 8's three new ideas (examples/demos/Demo8_Straight_Room_Twin/straight_room.hpp), with no Kinect.
//   1. Straighten: check_room's patterned box room photographed from 6 spots at their true poses; then scans 3-6 are
//      given a growing tilt (turned about their own camera, as a placing error handed on from scan to scan does) and
//      lifted 5-20 cm. Straighten must put every scan back within 0.3 deg and 2 cm, and leave the right ones alone.
//   2. FindCeilingMap on a made-up hall (points and facings written directly): a ceiling at 3.0 m over one half and
//      4.0 m over the other, a hole in the high part nobody saw, a lamp hanging at 2.6 m and one at 2.8 m in the hole (nothing
//      seen above it to outvote it). Each spot's height must be its own part's, the lamps' spots and the hole included;
//      CeilingMapPoints takes the ceiling's points and no wall's.
//   3. HangingPoints on that hall: both lamps hang; the walls and a tall cabinet (its top over the 2.1 m line) don't.
// Not covered: a real hall's sloped ceiling, beams built into a wall (they reach down the wall, so they stay), a room
// with steps in its floor (Straighten would set every scan on one floor).
// Run: build\check_straight_room.exe. Exit code 0 = passed. Writes nothing.

#include "../../examples/demos/Demo8_Straight_Room_Twin/straight_room.hpp"
#include "test_room.hpp"

#include <cstdio>

//double DegreesApart(const cv::Matx33f& a, const cv::Matx33f& b){
//
//@brief - the angle of the turn from a to b, in degrees. O(1).
double DegreesApart(const cv::Matx33f& a, const cv::Matx33f& b)
{
    const cv::Matx33f d = a.t() * b;
    return std::acos(std::clamp((d(0, 0) + d(1, 1) + d(2, 2) - 1) / 2.0, -1.0, 1.0)) * 180 / CV_PI;
}

//void CheckStraighten(){
//
//@brief - part 1: six shots of the test room, four of them tilted and lifted, straightened back. O(6 photographs).
void CheckStraighten()
{
    const cv::Matx33f firstCamera(1, 0, 0, 0, -1, 0, 0, 0, -1);   // the shots' coordinates (Y down) -> the room's (Y up)
    const spotter::Pose spots[6] = { Spot(0, 0, 0, 0, 20), Spot(0.5f, 0, 0.5f, 30, 20), Spot(-0.5f, 0, 0.3f, -40, 20),
                                     Spot(1.0f, 0, 0.8f, 70, 25), Spot(-0.8f, 0, 1.0f, -80, 25), Spot(0.3f, 0, 0.8f, 170, 20) };
    std::vector<spotter::Scan> scans;
    std::vector<spotter::Scan*> order;
    for (int s = 0; s < 6; s++) { scans.push_back(Photograph(spots[s], s + 1)); }
    double worstBefore = 0;
    for (int s = 0; s < 6; s++)
    {
        order.push_back(&scans[s]);
        const int k = std::max(0, s - 1);   // scans 1 and 2 placed right, then a growing error
        const cv::Matx33f wrong = spotter::TurnAbout(cv::Vec3d(1.2 * k * CV_PI / 180, 0, -1.8 * k * CV_PI / 180));
        const cv::Matx33f turn = wrong * (firstCamera * spots[s].turn);
        const cv::Point3f camera = firstCamera * spots[s].shift + cv::Point3f(0, 0.05f * k, 0);
        scans[s].pose.turn = firstCamera.t() * turn;
        scans[s].pose.shift = firstCamera.t() * camera;
        worstBefore = std::max(worstBefore, DegreesApart(scans[s].pose.turn, spots[s].turn));
    }
    const straight::Straightened result = straight::Straighten(order, firstCamera, straight::StraightenOptions());
    for (const std::string& line : result.report) std::printf("1. %s\n", line.c_str());
    double worstTurn = 0, worstShift = 0, worstRight = 0;
    int floors = 0;
    for (int s = 0; s < 6; s++)
    {
        const double degrees = DegreesApart(scans[s].pose.turn, spots[s].turn), shift = cv::norm(scans[s].pose.shift - spots[s].shift);
        std::printf("   shot %d: %.2f deg and %.1f cm from the truth after; floor %s\n", s + 1, degrees, shift * 100, result.fixes[s].floorSeen ? "seen" : "not seen");
        worstTurn = std::max(worstTurn, degrees);
        worstShift = std::max(worstShift, shift);
        if (s < 2) worstRight = std::max(worstRight, degrees);
        floors += result.fixes[s].floorSeen;
    }
    std::printf("   worst before %.2f deg; after %.2f deg, %.1f cm; floor %.3f m (true -1.200), seen by %d shots\n", worstBefore, worstTurn,
                worstShift * 100, result.floorY, floors);
    Check(worstBefore > 8, "the tilted shots really were tilted (the test means something)");
    Check(worstTurn < 0.3, "every shot stood up straight again, within 0.3 deg");
    Check(worstRight < 0.3, "the shots placed right stay right");
    Check(worstShift < 0.02, "every shot back at its height, within 2 cm");
    Check(std::abs(result.floorY + 1.2f) < 0.02f, "the room's floor at its true height");
    Check(floors >= 3, "the floor was seen and believed under at least 3 shots");
}

// The made-up hall (Y up, metres): floor y 0, x 0..8, z 0..6; ceiling 3.0 m for x < 4, 4.0 m for x >= 4 with an unseen
// hole x 6..7, z 2..3; walls all round up to the ceiling over them; a lamp x 5..5.6, z 1..1.3 at 2.6 m and one in the hole (underside
// and top); a cabinet against the x = 8 wall, z 4..5, 2.4 m tall, front at x 7.6.
struct Hall
{
    spotter::TwinRoom room;
    std::vector<int> kind;   // 0 floor, 1 ceiling, 2 wall, 3 lamp, 4 cabinet
};

//void Add(Hall& hall, cv::Point3f p, cv::Point3f n, int kind){
//
//@brief - one point of the hall. O(1).
void Add(Hall& hall, cv::Point3f p, cv::Point3f n, int kind)
{
    hall.room.points.push_back(p);
    hall.room.normals.push_back(n);
    hall.room.colours.push_back(cv::Vec3b(128, 128, 128));
    hall.kind.push_back(kind);
}

//Hall MakeHall(){
//
//@brief - the hall above, a point every 3 cm. O(points).
Hall MakeHall()
{
    Hall hall;
    const float d = 0.03f;
    auto CeilingAt = [](float x) { return x < 4 ? 3.0f : 4.0f; };
    for (float x = 0.01f; x < 8; x += d)
        for (float z = 0.01f; z < 6; z += d)
        {
            Add(hall, { x, 0, z }, { 0, 1, 0 }, 0);
            if (!(x >= 6 && x < 7 && z >= 2 && z < 3)) Add(hall, { x, CeilingAt(x), z }, { 0, -1, 0 }, 1);
        }
    for (float y = 0.01f; y < 4; y += d)
    {
        for (float z = 0.01f; z < 6; z += d)
        {
            if (y < 3) Add(hall, { 0, y, z }, { 1, 0, 0 }, 2);
            Add(hall, { 8, y, z }, { -1, 0, 0 }, 2);
            if (y > 3) Add(hall, { 4, y, z }, { -1, 0, 0 }, 2);   // the step up between the two ceilings
        }
        for (float x = 0.01f; x < 8; x += d)
            if (y < CeilingAt(x)) { Add(hall, { x, y, 0 }, { 0, 0, 1 }, 2); Add(hall, { x, y, 6 }, { 0, 0, -1 }, 2); }
    }
    for (float x = 5; x < 5.6f; x += d)
        for (float z = 1; z < 1.3f; z += d) { Add(hall, { x, 2.6f, z }, { 0, -1, 0 }, 3); Add(hall, { x, 2.65f, z }, { 0, 1, 0 }, 3); }
    for (float x = 6.4f; x < 6.8f; x += d)   // a second lamp, under the unseen hole: nothing above it to outvote it
        for (float z = 2.4f; z < 2.7f; z += d) { Add(hall, { x, 2.8f, z }, { 0, -1, 0 }, 3); Add(hall, { x, 2.85f, z }, { 0, 1, 0 }, 3); }
    for (float z = 4; z < 5; z += d)
    {
        for (float y = 0.01f; y < 2.4f; y += d) Add(hall, { 7.6f, y, z }, { -1, 0, 0 }, 4);
        for (float x = 7.6f; x < 8; x += d) Add(hall, { x, 2.4f, z }, { 0, 1, 0 }, 4);
    }
    return hall;
}

//void CheckCeilingAndHanging(){
//
//@brief - parts 2 and 3 on the made-up hall. O(points).
void CheckCeilingAndHanging()
{
    const Hall hall = MakeHall();
    const straight::CeilingMapOptions options;
    const straight::CeilingMap map = straight::FindCeilingMap(hall.room, 0, options);
    for (const std::string& line : map.report) std::printf("2. %s\n", line.c_str());
    const float low = map.At(2, 3), high = map.At(6, 1), overLamp = map.At(5.3f, 1.15f), overHole = map.At(6.1f, 2.1f), overLamp2 = map.At(6.6f, 2.55f);
    std::printf("   heights: low part %.2f m (true 3.00), high part %.2f m (4.00), over the lamp %.2f m (4.00), over the unseen hole %.2f m (4.00), "
                "over the lamp in it %.2f m (4.00)\n", low, high, overLamp, overHole, overLamp2);
    Check(map.seen, "the ceiling was found");
    Check(std::abs(low - 3) < 0.05f && std::abs(high - 4) < 0.05f, "each part of the ceiling has its own height");
    Check(std::abs(overLamp - 4) < 0.05f, "a lamp hanging lower is not taken for the ceiling");
    Check(std::abs(overHole - 4) < 0.05f, "an unseen part of the ceiling takes the height around it");
    Check(std::abs(overLamp2 - 4) < 0.05f, "a lamp with no ceiling seen above it is not taken for the ceiling either (a small lower piece)");

    const std::vector<char> isCeiling = straight::CeilingMapPoints(hall.room, map, options);
    int pieces = 0;
    const std::vector<char> hangs = straight::HangingPoints(hall.room, isCeiling, 0, pieces, straight::HangingOptions());
    int ceilingKept = 0, ceilingAll = 0, wrongCeiling = 0, lampHangs = 0, lampAll = 0, wrongHanging = 0;
    for (size_t i = 0; i < hall.kind.size(); i++)
    {
        const int kind = hall.kind[i];
        if (kind == 1) { ceilingAll++; ceilingKept += isCeiling[i]; }
        // a wall point under its own spot's ceiling (the step's face over the low part stands above that spot's ceiling)
        else if (isCeiling[i] && !(kind == 2 && hall.room.points[i].y > map.At(hall.room.points[i].x, hall.room.points[i].z))) wrongCeiling++;
        if (kind == 3) { lampAll++; lampHangs += hangs[i]; }
        else if (hangs[i]) wrongHanging++;
    }
    std::printf("3. ceiling points taken %d of %d, other points taken %d; lamp points hanging %d of %d, other points hanging %d (%d pieces)\n",
                ceilingKept, ceilingAll, wrongCeiling, lampHangs, lampAll, wrongHanging, pieces);
    Check(ceilingKept == ceilingAll, "every ceiling point is taken for the ceiling");
    Check(wrongCeiling == 0, "no wall, floor, lamp or cabinet point is taken for the ceiling");
    Check(lampHangs == lampAll && pieces == 2, "both lamps hang, each one piece");
    Check(wrongHanging == 0, "the walls and the cabinet (its top over the line) don't hang");
}

int main()
{
    CheckStraighten();
    CheckCeilingAndHanging();
    std::printf(failures ? "check_straight_room: %d FAILED\n" : "check_straight_room: all passed\n", failures);
    return failures ? 1 : 0;
}
