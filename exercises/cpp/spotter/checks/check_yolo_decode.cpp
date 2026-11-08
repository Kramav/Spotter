// Self-check for the YOLO decoding (detector_yolo_opencvdnn.hpp) and ClassList (detection.hpp),
// on made-up model outputs: no model file is loaded. Both output layouts, so a different model
// (an older export, or one you trained) decodes right before it ever runs.
// Run: build\check_yolo_decode.exe. Exit code 0 = passed.

#include "spotter/detector_yolo_opencvdnn.hpp"

#include <cstdio>

int failures = 0;
void Check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAILED: %s\n", what); failures++; }
}

bool Near(const cv::Rect2d& a, const cv::Rect2d& b) { return cv::norm(a.tl() - b.tl()) < 1e-3 && cv::norm(a.br() - b.br()) < 1e-3; }

int main()
{
    // A 1920 x 1080 picture in a 640 square: shrunk to a third (640 x 360), 140 px of grey above and below.
    spotter::Letterbox letterbox = spotter::Letterbox::For(cv::Size(1920, 1080), 640);
    Check(std::abs(letterbox.scale - 1.0 / 3) < 1e-9 && letterbox.padLeft == 0 && letterbox.padTop == 140, "letterbox");

    // 1. End-to-end (YOLO26, nms=False): 1 x 300 x 6 rows of x1 y1 x2 y2 confidence classId.
    int endToEndSize[] = { 1, 300, 6 };
    cv::Mat endToEnd(3, endToEndSize, CV_32F, cv::Scalar(0));
    const float rows[12] = { 100, 240, 200, 340, 0.9f, 62,    10, 150, 20, 160, 0.3f, 0 };
    std::copy(rows, rows + 12, endToEnd.ptr<float>());
    std::vector<spotter::Detection> found = spotter::DecodeYolo(endToEnd, spotter::YoloLayout::EndToEnd, letterbox, 0.4f);
    Check(found.size() == 1 && found[0].classId == 62 && Near(found[0].box, cv::Rect2d(300, 300, 300, 300)),
          "end-to-end: one box above the threshold, back in picture pixels");

    // 2. Classic: 1 x (4 + 3 classes) x 8400, one COLUMN per candidate: cx cy w h score0 score1 score2.
    //    Two class-1 boxes overlap (keep the better); a class-2 box on the same spot stays.
    int classicSize[] = { 1, 7, 8400 };
    cv::Mat classic(3, classicSize, CV_32F, cv::Scalar(0));
    cv::Mat table(7, 8400, CV_32F, classic.ptr<float>());
    auto Candidate = [&](int column, float cx, float cy, float w, float h, int classId, float score)
    {
        table.at<float>(0, column) = cx;
        table.at<float>(1, column) = cy;
        table.at<float>(2, column) = w;
        table.at<float>(3, column) = h;
        table.at<float>(4 + classId, column) = score;
    };
    Candidate(10, 150, 290, 100, 100, 1, 0.8f);
    Candidate(11, 152, 291, 100, 100, 1, 0.6f);
    Candidate(12, 150, 290, 100, 100, 2, 0.7f);
    Candidate(13, 500, 290, 50, 50, 0, 0.1f);   // below the threshold
    found = spotter::DecodeYolo(classic, spotter::YoloLayout::Classic, letterbox, 0.4f);
    int class1 = 0, class2 = 0;
    for (const spotter::Detection& d : found)
    {
        class1 += d.classId == 1;
        class2 += d.classId == 2;
        if (d.classId == 1)
            Check(std::abs(d.confidence - 0.8f) < 1e-6 && Near(d.box, cv::Rect2d(300, 300, 300, 300)), "classic: the better box, in picture pixels");
    }
    Check(found.size() == 2 && class1 == 1 && class2 == 1, "classic: duplicates removed per class only");

    // 3. Class names from the file next to the model.
    spotter::ClassList names(SPOTTER_REPO_ROOT "/yolo26n.names");
    Check(names.Size() == 80 && names.Name(0) == "person" && names.Name(62) == "tv", "yolo26n.names read");
    Check(names.Name(80) == "?" && names.Name(-1) == "?", "an id outside the list is \"?\", not a crash");
    Check(names.Find("person") == 0 && names.Find("unicorn") == -1, "find a class by name");

    std::printf(failures == 0 ? "PASSED\n" : "%d FAILED\n", failures);
    return failures == 0 ? 0 : 1;
}
