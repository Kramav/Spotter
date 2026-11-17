#include "spotter/text.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <opencv2/imgproc.hpp>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace spotter
{

#ifdef _WIN32

namespace
{
// One memory device context with one font per (pixel height, bold), kept for the program's life.
struct Gdi
{
    HDC dc = nullptr;
    std::map<std::pair<int, bool>, HFONT> fonts;

    Gdi() { dc = CreateCompatibleDC(nullptr); }

    HFONT Font(int height, bool bold)
    {
        auto key = std::make_pair(height, bold);
        auto found = fonts.find(key);
        if (found != fonts.end()) return found->second;
        // Arial Unicode MS; if it isn't installed Windows picks the nearest face (Arial).
        HFONT font = CreateFontW(-height, 0, 0, 0, bold ? FW_BOLD : FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_TT_PRECIS,
                                 CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, DEFAULT_PITCH | FF_SWISS, L"Arial Unicode MS");
        fonts[key] = font;
        return font;
    }
};

Gdi& Shared()
{
    static Gdi gdi;
    return gdi;
}

std::wstring Widen(const std::string& text)
{
    if (text.empty()) return std::wstring();
    int count = MultiByteToWideChar(CP_UTF8, 0, text.data(), (int)text.size(), nullptr, 0);
    std::wstring wide(count > 0 ? count : 0, L'\0');
    if (count > 0) MultiByteToWideChar(CP_UTF8, 0, text.data(), (int)text.size(), &wide[0], count);
    return wide;
}

int PixelHeight(double scale) { return std::max(8, (int)std::lround(30 * scale)); }
}

int TextWidth(const std::string& text, double scale, int thickness)
{
    Gdi& gdi = Shared();
    std::wstring wide = Widen(text);
    SelectObject(gdi.dc, gdi.Font(PixelHeight(scale), thickness >= 2));
    SIZE size{};
    GetTextExtentPoint32W(gdi.dc, wide.c_str(), (int)wide.size(), &size);
    return (int)size.cx;
}

void PutText(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar colour, int thickness)
{
    if (image.empty() || image.depth() != CV_8U || text.empty()) return;
    Gdi& gdi = Shared();
    std::wstring wide = Widen(text);
    SelectObject(gdi.dc, gdi.Font(PixelHeight(scale), thickness >= 2));
    TEXTMETRICW metrics{};
    GetTextMetricsW(gdi.dc, &metrics);
    SIZE size{};
    GetTextExtentPoint32W(gdi.dc, wide.c_str(), (int)wide.size(), &size);
    int width = size.cx + 4, height = metrics.tmHeight + 2;

    // White text on a black 32-bit bitmap = the coverage of each pixel.
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;   // top-down
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    HBITMAP bitmap = CreateDIBSection(gdi.dc, &info, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!bitmap || !bits) return;
    HGDIOBJ previous = SelectObject(gdi.dc, bitmap);
    cv::Mat mask(height, width, CV_8UC4, bits);
    mask.setTo(cv::Scalar(0, 0, 0, 0));
    SetBkMode(gdi.dc, TRANSPARENT);
    SetTextColor(gdi.dc, RGB(255, 255, 255));
    TextOutW(gdi.dc, 2, 1, wide.c_str(), (int)wide.size());
    GdiFlush();

    // Blend the colour in through the coverage; the text's top-left is the baseline origin moved up by the ascent.
    cv::Point topLeft(origin.x - 2, origin.y - metrics.tmAscent - 1);
    cv::Rect area = cv::Rect(topLeft, mask.size()) & cv::Rect(0, 0, image.cols, image.rows);
    if (!area.empty())
    {
        int channels = std::min(image.channels(), 3);
        for (int y = 0; y < area.height; y++)
        {
            const cv::Vec4b* from = mask.ptr<cv::Vec4b>(y + area.y - topLeft.y) + (area.x - topLeft.x);
            uchar* to = image.ptr<uchar>(y + area.y) + area.x * image.channels();
            for (int x = 0; x < area.width; x++, to += image.channels())
            {
                int cover = from[x][1];   // green channel: no colour fringes whatever GDI did
                if (cover == 0) continue;
                for (int c = 0; c < channels; c++)
                    to[c] = (uchar)((to[c] * (255 - cover) + (int)colour[channels == 1 ? 0 : c] * cover + 127) / 255);
            }
        }
    }
    SelectObject(gdi.dc, previous);
    DeleteObject(bitmap);
}

#else

int TextWidth(const std::string& text, double scale, int thickness)
{
    return cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, nullptr).width;
}

void PutText(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar colour, int thickness)
{
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv::LINE_AA);
}

#endif

}
