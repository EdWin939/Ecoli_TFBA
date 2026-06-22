// === PATHS ===

pre = "/Users/eduardmrug/Documents/Honours research/Labwork/Experiments/20250729_MWT305/";
inputDir  = pre + "original_tiff/";
outputDir = pre + "tifffiles/";
outputPrefix = "20250729_MWT305_";
ext = ".tif";

// === START SCRIPT ===
list = getFileList(inputDir);
print("📁 Input folder: " + inputDir);
print("📄 Found " + list.length + " files");

for (i = 0; i < list.length; i++) {
    name = list[i];

    if (endsWith(name, "_bf" + ext)) {
        baseName = replace(name, "_bf" + ext, "");
        print("🔍 Processing base: " + baseName);

        fileBF  = inputDir + baseName + "_bf"  + ext;
        fileCFP = inputDir + baseName + "_cfp" + ext;
        fileYFP = inputDir + baseName + "_yfp" + ext;

        if (File.exists(fileCFP) && File.exists(fileYFP)) {
            print("✅ Found all channels for " + baseName);

            // Open and store window titles
            run("Bio-Formats Importer", "open=[" + fileBF + "] autoscale color_mode=Default view=Hyperstack stack_order=Default");
            bfTitle = getTitle();

            run("Bio-Formats Importer", "open=[" + fileCFP + "] autoscale color_mode=Default view=Hyperstack stack_order=Default");
            cfpTitle = getTitle();

            run("Bio-Formats Importer", "open=[" + fileYFP + "] autoscale color_mode=Default view=Hyperstack stack_order=Default");
            yfpTitle = getTitle();

            // Make sure all opened
            if (isOpen(bfTitle) && isOpen(cfpTitle) && isOpen(yfpTitle)) {
                selectWindow(bfTitle);
                selectWindow(cfpTitle);
                selectWindow(yfpTitle);

                run("Images to Stack", "name=" + baseName + "_stack title=[] use");

                // Save the stack
                suffix = substring(baseName, lengthOf(baseName)-2); // gets "001" from "image001"
		saveAs("Tiff", outputDir + outputPrefix + suffix + ".tiff");
                print("💾 Saved: " + outputPrefix + baseName + ".tiff");
            } else {
                print("⚠️ One or more images failed to open for " + baseName);
            }

            run("Close All");
        } else {
            print("⚠️ Skipping " + baseName + " — missing CFP or YFP file.");
        }
    }
}