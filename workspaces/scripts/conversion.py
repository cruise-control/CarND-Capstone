from random import shuffle
import pandas as pd
import yaml

# Process a line from the original labeled annotation file
def processLine(line):
    # Split this segment by ,
    res = line.split(',')
    output = []
    # Go over every output
    for i, r in enumerate(res):
        # The first output is discarded as it has no useful data
        if i == 0:
            continue
        # Split by : to extract name and value
        rr = r.split(':')
        # Strip out any unwanted characters and append to output list
        output.append(rr[1].strip('}'))
    return output

def convertToDict(boxes,f, label):
    x = float(box[0])
    y = float(box[1])
    w = float(box[2])
    h = float(box[3])
    x_min = x #int(x - w/2)
    x_max = x + w #int(x + w/2)
    y_min = y #int(y - h/2)
    y_max = y + h #int(y + h/2)
    
    path = {}
    path['path'] = f
    boxes = {}
    #boxes['boxes'] = []
    
    entry = {}
    entry['label'] = label
    entry['occluded'] = False
    entry['xmax'] = x_max
    entry['xmin'] = x_min
    entry['ymax'] = y_max
    entry['ymin'] = y_min
    
    boxes['boxes'] = entry
    
    return boxes, path

# Simple value clamp on the bounding boxes
def clamp(value, mx):
    return max(min(value, mx*0.999), mx*0.001)
    
# Convert an entry to a CSV string
def convertToCSV(boxes,f, label, header=False):
    if header:
        return 'filename,width,height,class,xmin,ymin,xmax,ymax'
    x = int(boxes[0])
    y = int(boxes[1])
    w = int(boxes[2])
    h = int(boxes[3])
    x_min = x 
    x_min = int(clamp( x, 800))
    x_max = int(clamp( x + w , 800)) 
    y_max = int(clamp( y + h , 600))
    y_min = int(clamp( y, 600)) 
        
    return str(f) + ',' + str(800) + ',' + str(600) + ',' + str(label) + ',' + str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + '\n'
    
# The data stripping and collection pipeline
def pipeline(filepath, prefix):
    dataframe = pd.read_csv(filepath)
    filename = dataframe['#filename']
    box_info = dataframe['region_shape_attributes']
    labels = dataframe['region_attributes']
    
    processed = []
    processed.append([processLine(x) for x in box_info])

    # Strip the unwanted characters
    labels = [x.strip('{') for x in labels]
    labels = [x.strip('}') for x in labels]
    labels = [x.split(':')[0] for x in labels]
    labels = [x.strip('"') for x in labels]
    
    shuffleList = []
    for box, f, l in zip(processed[0], filename, labels):
        shuffleList.append((box,f,l))
        
    shuffle(shuffleList)
    
    entry = ''
    for box, f, l in shuffleList: #zip(processed[0], filename, labels):
        entry += convertToCSV(box,prefix + '/' + f,l, header=False)
        
    return entry
    
prefix = ''
# Create a CSV formatted string with all of the information
output = convertToCSV(None,None,None, header=True) + '\n'
output += pipeline('./Simulation/via_region_data.csv', './Simulation')
output += pipeline('./RosBag/bag_region_data.csv', './RosBag')
print(output)
      
# Write it to a csv file
with open('compiled_traffic_lights.csv', 'w') as the_file:
    the_file.write(output)



