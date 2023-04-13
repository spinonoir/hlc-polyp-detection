import os
import xmltodict
import shutil


def edit_xml_file(data, fname):
    back_string = """
        <object>
            <name>background</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>1</xmin>
                <ymin>1</ymin>
                <xmax>3</xmax>
                <ymax>3</ymax>
            </bndbox>
        </object>    
    """
    backxml = xmltodict.parse(back_string)

    e = xmltodict.parse(data)
    doc = e['annotation']
    
    # Change the filename to flattened name with correct prefix and extension:
    doc['filename'] = fname

    obj = doc.get('object')

    if not obj:
        doc['object'] = backxml['object']
    else:
        if type(doc['object']) is list:
            for obj in doc['object']:
                obj['name'] = 'polyp'
        else:
            doc['object']['name'] = 'polyp'

    

    # If the current 
    xmlstr = xmltodict.unparse(e)
    return xmlstr


def flatten_files(source_directory = 'PolypsSet'):
    """"
    Flatten the directory structure of the PolypsSet dataset, renaming the files and updating the XML files.
    
    Directory structure expected:
    
    PolypsSet/
    ├── test/
    │   ├── Annotation/
    │   │   ├── N/
    │   │   │   ├── M.xml
    │   │   │   ├── ... .xml
    │   │   ├── .../
    │   └── Image/
    │       ├── N/
    │       │   ├── M.jpg
    │       │   ├── ... .jpg
    │       ├── .../
    ├── train/
    │   ├── Annotation/
    │   │   ├── M.xml
    │   │   ├── ... .xml
    │   └── Image/
    │       ├── M.jpg
    │       ├── ... .jpg  
    └── val/
        ├── Annotation/
        │   ├── N/
        │   │   ├── M.xml
        │   │   ├── ... .xml
        │   ├── .../
        └── Image/
            ├── N/
            │   ├── M.jpg
            │   ├── ... .jpg
            ├── .../  

    flattened image:  val/Image/N/M.jpg      --> val_seqN_frameM.jpg
    flattened xml  :  val/Annotation/N/M.xml --> val_seqN_frameM.xml

    XML contents are also updated:
      - all polyps classified as 'polyp'
      - filename is updated to match the flattened name
      - images without polyps are labeled 'background'
      - images without polyps are assigned a null bounding box
    """
    def update_files(frames, f_name, sub_dir, bottom_level, count):
        # Ignore stupid .DS_Store files
        if frames[0].startswith('.'): frames.pop(0)
        for x in frames:
            x_name = f_name + '_frame' + x 
            x_path_src = os.path.join(bottom_level, x)                               #PolypsSet/[test or val]/[Image or Annotation]/[N]/[M.xml or M.jpg]  

            # assert the path exists
            assert os.path.exists(x_path_src), "The path does not exist."

            # trgt is the full path to where the file will be moved
            trgt = os.path.join(source_directory, sub_dir, x_name)                  #PolypsSet/[test or val]/[[test or val]_seqN_frameM.[jpg or xml]

            # Edit the XML file
            if x.endswith('.xml'):
                # update the xml file
                img_name = x_name[:-4] + '.jpg'
                with open(x_path_src, 'r+') as f:
                    data = f.read()
                    fixed_data = edit_xml_file(data, img_name)
                    # if(count % 10000 == 0):
                    #     print(fixed_data)
                    f.seek(0)
                    f.write(fixed_data)
                    f.truncate()
            count += 1
            if(count % 100 == 0):
                # Sanity check: print the file name and the target path to the console
                    print(f'{x_path_src} --> {trgt}')
            # Rename and move the file 
            os.rename(x_path_src, trgt)


    # assert the TLD directory exists
    assert os.path.exists(source_directory), "The source directory does not exist."

    # Get the list of directories in the TLD directory
    dirs = os.listdir(source_directory)

    # Ignore stupid.DS_Store files
    if dirs[0].startswith('.'): dirs.pop(0)

    count = 0
    for sub_dir in dirs:
        level_one = os.path.join(source_directory, sub_dir)                                #PolypsSet/[test or val or train]

        # assert the directory exists
        assert os.path.exists(level_one), "The directory does not exist." 
        sub_sub_dirs = os.listdir(level_one)                                                     

        # Ignore stupid .DS_Store files
        if sub_sub_dirs[0].startswith('.'): sub_sub_dirs.pop(0)
        for sub_sub_dir in sub_sub_dirs:  
            # sub_sub_dir is either 'Image' or 'Annotation'
            level_two = os.path.join(level_one, sub_sub_dir)                                     #PolypsSet/[test or val or train]/[Image or Annotation]

            # assert the directory exists
            assert os.path.exists(level_two), "The directory does not exist."
            # Training set has no subdirectories, so we need to handle it differently
            if sub_dir != 'train':
                sequences = os.listdir(level_two)
                # Ignore stupid .DS_Store files
                if sequences[0].startswith('.'): sequences.pop(0)
                for sequence in sequences:
                    bottom_level = os.path.join(level_two, sequence)                             #PolypsSet/[test or val]/[Image or Annotation]/[N]
                    f_name = sub_dir + '_seq' + sequence                                       #[test or val]_seqN
                     # assert the directory exists
                    assert os.path.exists(bottom_level), "The directory does not exist."
                    frames = os.listdir(bottom_level)
                    update_files(frames, f_name, sub_dir, bottom_level, count)     
            else:
                bottom_level = level_two
                f_name = sub_dir
                 # assert the directory exists
                assert os.path.exists(bottom_level), "The directory does not exist."
                frames = os.listdir(bottom_level)
                update_files(frames, f_name, sub_dir, bottom_level, count)

                                          

           

    # Remove the empty directories
    # for sub_dir in dirs:
    #     level_one = os.path.join(source_directory, sub_dir)
    #     sub_sub_dirs = os.listdir(level_one)
    #     if sub_sub_dirs[0].startswith('.'): sub_sub_dirs.pop(0)
    #     for sub_sub_dir in sub_sub_dirs:  
    #         level_two = os.path.join(level_one, sub_sub_dir)
    #         if sub_dir != 'train':
    #             sequences = os.listdir(level_two)
    #             if sequences[0].startswith('.'): sequences.pop(0)
    #             for sequence in sequences:
    #                 bottom_level = os.path.join(level_two, sequence)
    #                 os.rmdir(bottom_level)
    #         else:
    #             bottom_level = level_two
    #         os.rmdir(bottom_level)
    #     os.rmdir(level_two)
        


if __name__ == '__main__':
    flatten_files()