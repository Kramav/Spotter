// spotter/text.hpp: text in pictures, in Arial Unicode MS (the demos' and examples' one font).
//
// OpenCV's own cv::putText only has the Hershey stroke fonts: ASCII only, thin, and a look of its own.
// This draws real Arial Unicode through the Windows GDI (no extra library; OpenCV's FreeType module isn't
// in our vcpkg build). Without Windows it falls back to cv::putText, so the programs still build and run.
// The `scale` and `thickness` mean what cv::putText's did (scale 1.0 is about 30 px tall), so a call
// changes by its name only. Text is UTF-8.
#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace spotter
{

//void PutText(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar colour, int thickness = 1){
//
//@param1 - image: an 8-bit 1-, 3- or 4-channel picture; drawn on in place
//@param2 - text: UTF-8
//@param3 - origin: the left end of the text's baseline, as in cv::putText
//@param4 - scale: 1.0 is about 30 px tall
//@param5 - colour: BGR(A), as everywhere in OpenCV
//@param6 - thickness: 2 or more draws bold
//
//@brief - draws the text, anti-aliased; parts outside the picture are cut off. O(text area).
void PutText(cv::Mat& image, const std::string& text, cv::Point origin, double scale, cv::Scalar colour, int thickness = 1);

//int TextWidth(const std::string& text, double scale, int thickness = 1){
//
//@brief - the text's width in pixels as PutText draws it (for wrapping and right-aligning). O(text).
int TextWidth(const std::string& text, double scale, int thickness = 1);

}
