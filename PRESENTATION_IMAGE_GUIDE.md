# Presentation Image Guide

## How to Add UI Screenshots to PowerPoint

This guide explains how to capture and add user interface screenshots to make the presentation more interactive and visual.

## Slides Requiring Screenshots

### Slide 8: User Interface - Main Screen
**What to capture:**
- Full Streamlit interface showing:
  - Main title "PDF QA System - Query Interface"
  - Query input box
  - Answer display area (can be empty for initial state)
  - Sidebar with system status

**How to capture:**
1. Run: `streamlit run app_new.py`
2. Wait for browser to open
3. Take full-screen screenshot (Windows: Win+Shift+S, Mac: Cmd+Shift+4)
4. Crop to show main interface area

### Slide 9: User Interface - Sidebar Features
**What to capture:**
- Sidebar showing:
  - System Status section (✅ System Ready)
  - Chunks count metric
  - Embeddings shape (e.g., "150 x 384")
  - "Trained Files" expandable section
  - Query Settings section
  - Top-K Results slider
  - PDF file selector (multiselect dropdown)

**How to capture:**
1. Expand the "Trained Files" section
2. Take screenshot of sidebar only
3. Ensure all metrics and settings are visible

### Slide 10: User Interface - Query & Results
**What to capture:**
- Complete query example showing:
  - Question in input box: "What is the main topic of this document?"
  - Generated answer below
  - Retrieved chunks with text content
  - Similarity scores (e.g., 0.88, 0.85, 0.82)
  - Source document names
  - Formatted output with sections

**How to capture:**
1. Enter a question in the query box
2. Click search/submit
3. Wait for results to appear
4. Take screenshot of full results area

### Slide 18: Live Demo - Example 1
**What to capture:**
- Query: "What is artificial intelligence?"
- Full interface with:
  - Question displayed
  - Answer with multiple chunks
  - High similarity scores
  - Source document references

**How to capture:**
1. Enter the example question
2. Submit query
3. Capture full result view

### Slide 19: Live Demo - Example 2
**What to capture:**
- Query: "How does the system process documents?"
- Different type of question showing:
  - Process-related answer
  - Technical explanation chunks
  - Multiple source documents
  - Detailed information

**How to capture:**
1. Enter the second example question
2. Submit query
3. Capture result showing technical details

## Steps to Add Screenshots

1. **Open the PowerPoint presentation**
   - File: `PDF_QA_Chatbot_Presentation.pptx`

2. **Navigate to the slide** (e.g., Slide 8)

3. **Delete the placeholder text box** that says "[SCREENSHOT PLACEHOLDER]"

4. **Insert the image:**
   - Go to Insert → Pictures → This Device
   - Select your screenshot
   - Resize to fit the placeholder area (approximately 8" wide x 4.5" tall)

5. **Position the image:**
   - Align to center of slide
   - Ensure it's below the title
   - Leave space for any notes below

6. **Optional enhancements:**
   - Add border: Right-click image → Format Picture → Picture Border
   - Add shadow for depth
   - Crop if needed to focus on important areas

## Tips for Better Screenshots

1. **Use high resolution:** Capture at full browser window size
2. **Clean interface:** Close unnecessary browser tabs/windows
3. **Good examples:** Use questions that return clear, relevant results
4. **Consistent styling:** Use same browser and zoom level for all screenshots
5. **Highlight important areas:** Use PowerPoint's annotation tools to circle key features

## Alternative: Using Screenshot Tools

### Windows:
- **Snipping Tool** or **Snip & Sketch** (Win+Shift+S)
- **Greenshot** (free, more features)
- **ShareX** (advanced screenshot tool)

### Mac:
- **Cmd+Shift+4** (select area)
- **Cmd+Shift+3** (full screen)
- **Screenshot app** (built-in)

### Browser Extensions:
- **Full Page Screen Capture** (Chrome)
- **FireShot** (Firefox/Chrome)

## Recommended Screenshot Settings

- **Format:** PNG (best quality) or JPG (smaller file size)
- **Resolution:** At least 1920x1080 for clarity
- **File size:** Keep under 2MB per image for presentation performance
- **Naming:** Use descriptive names like "ui_main_screen.png", "ui_sidebar.png"

## After Adding Images

1. **Test the presentation** to ensure images display correctly
2. **Check file size** - if presentation is too large, compress images
3. **Verify readability** - text in screenshots should be readable
4. **Add animations** (optional) - use PowerPoint animations to reveal images

## Making It Interactive

### Add Animations:
1. Select image
2. Go to Animations tab
3. Choose entrance effect (e.g., "Fade In", "Fly In")
4. Set timing

### Add Transitions:
1. Select slide
2. Go to Transitions tab
3. Choose transition effect (e.g., "Fade", "Push")
4. Apply to all slides for consistency

### Add Hyperlinks (Optional):
- Link demo slides to actual running application
- Add links to documentation
- Link to GitHub repository

## Final Checklist

- [ ] All 5 screenshot placeholders replaced
- [ ] Images are clear and readable
- [ ] Images are properly sized and aligned
- [ ] Presentation file size is reasonable (< 50MB)
- [ ] Animations/transitions added (optional)
- [ ] Presentation tested on different computers
- [ ] All slides have consistent styling

## Quick Reference

| Slide | Content | Screenshot Type |
|-------|---------|----------------|
| 8 | Main Interface | Full UI view |
| 9 | Sidebar | Sidebar only |
| 10 | Query Results | Query + Results |
| 18 | Demo Example 1 | Full demo view |
| 19 | Demo Example 2 | Full demo view |

---

**Note:** The presentation is fully functional without screenshots, but adding them will make it much more engaging and professional!

