// spotter/file_dialog.hpp: asking the user, with the system's own windows (D48).
//
//   AskSaveAs      the standard "Save As" window (13e: the room folder at start-up and on K, the .obj on E)
//   AskOpenFolder  the standard "Select Folder" window (13w, 13x, 13y, 13z, Demos 4 and 6: which room)
//   Confirm        a Yes / No box (No is the default)
//   Tell           a message box that has to be clicked away: for failures, so they can't be missed
//   Ask            one line typed in the console
//
// Windows: the shell's dialogs (IFileSaveDialog, IFileOpenDialog) and MessageBox. Linux: zenity, when there
// is a desktop. Otherwise, or when that fails, or with SPOTTER_CONSOLE_DIALOGS=1 in the environment, the
// console asks instead. A relative answer typed there is taken from the suggested folder, never from the
// folder the program happens to run in (the cause of D48: a room saved inside build/ by accident).
// Every dialog also prints what it asked and the answer to the console, so the console keeps a record.
// The dialogs are modal: the caller's loop waits until they close.
#pragma once

#include <filesystem>
#include <string>

namespace spotter
{

//std::filesystem::path AskSaveAs(const std::string& title, const std::filesystem::path& folder, const std::string& name, const std::string& filterName, const std::string& extension){
//
//@param1 - title: the window's title, e.g. "13e: save the room as"
//@param2 - folder: where it opens (if it doesn't exist yet, the nearest folder above it that does)
//@param3 - name: the suggested file name
//@param4 - filterName: what's being saved, e.g. "Wavefront OBJ"
//@param5 - extension: e.g. "obj": added when missing, and an existing file is overwritten only after a yes.
//          Empty = a folder name is being chosen: only folders are listed and nothing is added.
//
//@brief - the full path chosen in a Save As window, or empty if it was cancelled. O(1) plus the wait.
std::filesystem::path AskSaveAs(const std::string& title, const std::filesystem::path& folder, const std::string& name,
                                const std::string& filterName, const std::string& extension);

//std::filesystem::path AskOpenFolder(const std::string& title, const std::filesystem::path& startIn, const std::string& preselect){
//
//@param1 - title: the window's title, e.g. "13w: open a room folder"
//@param2 - startIn: where it opens (or the nearest folder above it that exists)
//@param3 - preselect: a folder name inside startIn to suggest (default none)
//
//@brief - the full path of the folder chosen, or empty if it was cancelled. O(1) plus the wait.
std::filesystem::path AskOpenFolder(const std::string& title, const std::filesystem::path& startIn, const std::string& preselect = "");

//bool Confirm(const std::string& title, const std::string& text){
//
//@param1 - title: the box's title
//@param2 - text: the question
//
//@brief - true only if Yes was chosen (No is the default; closing the box is No). O(1) plus the wait.
bool Confirm(const std::string& title, const std::string& text);

//void Tell(const std::string& title, const std::string& text){
//
//@param1 - title: the box's title
//@param2 - text: the message (a failure and what to do about it)
//
//@brief - prints the message to the console and shows it in a box that has to be clicked away. O(1) plus the wait.
void Tell(const std::string& title, const std::string& text);

//std::string Ask(const std::string& question){
//
//@param1 - question: printed before waiting
//
//@brief - a line typed in the console, without spaces or quotes round it ("Copy as path" adds quotes).
//         "cancel" at the end of input. O(answer length).
std::string Ask(const std::string& question);

}
