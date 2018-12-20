% FILE FORMATS FOR THE MNIST DATABASE
% see http://yann.lecun.com/exdb/mnist/
% Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
% however, imshow is the opposite: 0 means black, 255 means white

clc;clear;close all;
%%
% test image
test_image=zeros(28,28,10000);
filename='t10k-images.idx3-ubyte';
fid = fopen(filename, 'r');
magic_number_temp = uint8(fread(fid, 4, 'uchar'));  % 50855936=ox00000803; 2051=0x03080000
magic_number=typecast(flipud(magic_number_temp),'int32');
if magic_number==2051
    number_of_image_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_image=typecast(flipud(number_of_image_temp),'int32');
    
    number_of_rows_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_rows = typecast(flipud(number_of_rows_temp),'int32');
    
    number_of_columns_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_columns = typecast(flipud(number_of_columns_temp),'int32');
    
    for i=1:number_of_image
        data=fread(fid, number_of_rows*number_of_columns, 'uchar');
        test_image(:,:,i)=reshape(data,number_of_rows,number_of_columns)';
    end
end
figure
for i=1:25
    subplot(5,5,i)
    imshow(test_image(:,:,i))
end
suptitle('test image')
fclose(fid);

% train image
train_image=zeros(28,28,60000);
filename='train-images.idx3-ubyte';
fid = fopen(filename, 'r');
magic_number_temp = uint8(fread(fid, 4, 'uchar'));  % 50855936=ox00000803; 2051=0x03080000
magic_number=typecast(flipud(magic_number_temp),'int32');
if magic_number==2051
    number_of_image_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_image=typecast(flipud(number_of_image_temp),'int32');
    
    number_of_rows_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_rows = typecast(flipud(number_of_rows_temp),'int32');
    
    number_of_columns_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_columns = typecast(flipud(number_of_columns_temp),'int32');
    
    for i=1:number_of_image
        data=fread(fid, number_of_rows*number_of_columns, 'uchar');
        train_image(:,:,i)=reshape(data,number_of_rows,number_of_columns)';
    end
end
figure
for i=1:25
    subplot(5,5,i)
    imshow(train_image(:,:,i))
end
suptitle('train image')
fclose(fid);

% test label
filename='t10k-labels.idx1-ubyte';
fid = fopen(filename, 'r');
magic_number_temp = uint8(fread(fid, 4, 'uchar'));  % 50855936=ox00000803; 2051=0x03080000
magic_number=typecast(flipud(magic_number_temp),'int32');
if magic_number==2049
    number_of_item_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_item=typecast(flipud(number_of_item_temp),'int32');
    
    test_label=fread(fid, number_of_item, 'uchar');
end
fclose(fid);

% train label
filename='train-labels.idx1-ubyte';
fid = fopen(filename, 'r');
magic_number_temp = uint8(fread(fid, 4, 'uchar'));  % 50855936=ox00000803; 2051=0x03080000
magic_number=typecast(flipud(magic_number_temp),'int32');
if magic_number==2049
    number_of_item_temp = uint8(fread(fid, 4, 'uchar'));
    number_of_item=typecast(flipud(number_of_item_temp),'int32');
    
    train_label=fread(fid, number_of_item, 'uchar');
end
fclose(fid);

save('MNIST_data.mat','test_image','train_image','test_label','train_label')
