/** 
 * Vakil Takhaveev, PhD student, MSB, RUG
 * v.takhaveev@rug.nl
 * 18.04.2017
 * José Losa
 * 24.11.2020
 * 21.10.2021 - updated to do background subtraction on BF images
 * 03.11.2021 - updated radius for background subtraction (50 --> 20)
 * 25.05.2025 - updated first step: starts with .nd2 stack and saves as individual .tif
 *
 *
 * This script:
 *  - opens each .nd2 stack in 'nd2stack' folder
 *  - splits it into separate channel .tif files (_bf, _cfp, _yfp) saved in 'split_images'
 *  - subtracts background (r=20) for BF images
 *  - saves processed images in 'original_tiff'
 */

inputpath = "/Users/eduardmrug/Documents/Honours research/Labwork/Experiments/20250729_MKO305/nd2files/";
split_path = "/Users/eduardmrug/Documents/Honours research/Labwork/Experiments/20250729_MKO305/splitimages/";
output_path = "/Users/eduardmrug/Documents/Honours research/Labwork/Experiments/20250729_MKO305/original_tiff/";

function splitAndSaveChannels(filename) {
    file_path = inputpath + filename;
    
    // Open the .nd2 stack
    run("Bio-Formats Importer", "open=["+file_path+"] autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
    
    // Split channels
    run("Split Channels");
    
    // Save each channel as .tif
    for (c=0; c<3; c++) {
        channelTitle = getTitle();
        if (c == 0) suffix = "_bf";
        else if (c == 1) suffix = "_cfp";
        else if (c == 2) suffix = "_yfp";
        
        new_filename = replace(filename, "\\.nd2", "") + suffix + ".tif";
        saveAs("Tiff", split_path + new_filename);
        close();
    }
}

function processFile(filename) {
    file_path = split_path + filename;
    
    open(file_path);
    
    filename_out = replace(filename, "\\.tif", "");
    
    // Subtract background for BF images
    if (indexOf(filename_out, "_bf") >= 0) {
        run("Subtract Background...", "rolling=20 light");
    	
    	
    }
    
    // Save as .tif in final output folder
    selectWindow(filename_out + ".tif");
    saveAs("Tiff", output_path + filename_out + ".tif");
    close();
}

// === Main script ===
list = getFileList(inputpath);

for (i = 0; i < list.length; i++) {
    if (endsWith(list[i], ".nd2")) {
        // 1) Split and save channels as .tif
        splitAndSaveChannels(list[i]);
    }
}

// Now process the individual channel .tif files
splitList = getFileList(split_path);

for (j = 0; j < splitList.length; j++) {
    if (endsWith(splitList[j], ".tif")) {
        processFile(splitList[j]);
    }
}

