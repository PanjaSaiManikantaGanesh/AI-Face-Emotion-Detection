% CNN Training Script for FER-2013 (Fixed for Grayscale Images)
% Save this as train_cnn.m

clc; clear; close all;

% Load dataset
datasetPath = 'fer2013';
imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Resize + ensure RGB format
imds.ReadFcn = @(x) preprocessImage(x);

% Split dataset (80% train / 20% test)
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Load pretrained network
net = vgg16;

% Modify the last 3 layers for our dataset
numClasses = numel(categories(imdsTrain.Labels));
layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 8, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
trainedCNN = trainNetwork(imdsTrain, layers, options);

% Save the trained model
save trainedCNN.mat trainedCNN
disp('✅ CNN training completed successfully and saved as trainedCNN.mat');


%% --- Helper function: ensures image is RGB and 224x224
function Iout = preprocessImage(filename)
    I = imread(filename);
    I = imresize(I, [224 224]);
    if size(I, 3) == 1
        I = cat(3, I, I, I); % Convert grayscale → RGB
    end
    Iout = I;
end

%% ----- Evaluate Model on Test Data -----

% Classify test set
YPred = classify(trainedCNN, imdsTest);
YTrue = imdsTest.Labels;

%% ----- Confusion Matrix -----
figure;
cm = confusionchart(YTrue, YPred);
cm.Title = 'Confusion Matrix - FER2013 CNN';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

%% ----- Classification Report -----
classes = categories(YTrue);
numClasses = numel(classes);

precision = zeros(numClasses,1);
recall = zeros(numClasses,1);
f1score = zeros(numClasses,1);

for i = 1:numClasses
    class = classes{i};
    
    TP = sum((YPred == class) & (YTrue == class));
    FP = sum((YPred == class) & (YTrue ~= class));
    FN = sum((YPred ~= class) & (YTrue == class));
    
    precision(i) = TP / (TP + FP + eps);
    recall(i)    = TP / (TP + FN + eps);
    f1score(i)   = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i)+eps);
end

overallAccuracy = mean(YPred == YTrue) * 100;

T = table(classes, precision, recall, f1score);
disp('----- Classification Report -----');
disp(T);

fprintf('\nOverall Accuracy: %.2f%%\n', overallAccuracy);




function face_expression_gui
    % Facial Expression Detection GUI - FER2013 (Image & Webcam)
    clc; clear; close all;
    warning off;

    % Main GUI Window
    hFig = figure('Name','Facial Expression Detection - FER2013',...
        'NumberTitle','off','Position',[400 150 720 520],...
        'Color',[0.9 0.9 0.9],'MenuBar','none','Resize','off');

    uicontrol('Style','text','String','Facial Expression Detection (FER-2013)',...
        'Position',[100 470 520 30],'FontSize',14,'FontWeight','bold');

    % Image axes
    hAxes = axes('Parent',hFig,'Units','pixels','Position',[260 120 420 320]);
    axis off;

    % Buttons
    uicontrol('Style','pushbutton','String','1. Image Selection',...
        'Position',[40 400 170 35],'FontSize',10,'Callback',@image_selection);

    
    uicontrol('Style','pushbutton','String','2. Face Recognition (Select Model)',...
        'Position',[40 350 170 35],'FontSize',10,'Callback',@face_recognition);

    uicontrol('Style','pushbutton','String','3. Webcam Detection (Real-Time)',...
        'Position',[40 300 170 35],'FontSize',10,'Callback',@webcam_detection);

    uicontrol('Style','pushbutton','String','4. Program Info',...
        'Position',[40 250 170 35],'FontSize',10,'Callback',@program_info);

    uicontrol('Style','pushbutton','String','5. Exit',...
        'Position',[40 200 170 35],'FontSize',10,'Callback',@exit_program);

    % Handles
    handles.img = [];
    guidata(hFig, handles);

    %% --- Callback Functions ---

    % -------------------------
    % IMAGE SELECTION
    % -------------------------
    function image_selection(~,~)
        [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select Image');
        if isequal(file,0), return; end
        img = imread(fullfile(path,file));
        axes(hAxes); imshow(img); title('Selected Image');
        handles.img = img;
        guidata(hFig, handles);
    end

    % -------------------------
    % ADD TO DATABASE
    % -------------------------
    function add_database(~,~)
        handles = guidata(hFig);
        if isempty(handles.img)
            msgbox('Please select an image first!','Warning','warn'); return;
        end
        answer = inputdlg('Enter Expression Label (Happy, Sad, etc):','Add to Database');
        if isempty(answer), return; end
        label = strtrim(answer{1});
        savePath = fullfile('database',label);
        if ~exist(savePath,'dir'), mkdir(savePath); end
        filename = fullfile(savePath,['img_' datestr(now,'HHMMSS') '.jpg']);
        imwrite(handles.img, filename);
        msgbox('Image added to database successfully!','Success');
    end

    % -------------------------
    % IMAGE-BASED CNN RECOGNITION
    % -------------------------
    function face_recognition(~,~)
        handles = guidata(hFig);
        if isempty(handles.img)
            msgbox('Please select an image first!','Warning','warn'); return;
        end

        if ~exist('trainedCNN.mat','file')
            msgbox('CNN model not found! Train using train_cnn.m first.','Error','error'); 
            return;
        end

        data = load('trainedCNN.mat');  
        trainedCNN = data.trainedCNN;

        img = imresize(handles.img, [224 224]);
        [label, scores] = classify(trainedCNN, img);

        axes(hAxes); imshow(handles.img);
        title(['Detected (CNN): ', char(label)]);

        msgbox(sprintf('Detected Expression: %s\nConfidence: %.2f%%',...
            char(label), max(scores)*100), 'CNN Result');
    end

    % -------------------------
    % REAL-TIME WEBCAM DETECTION
    % -------------------------
  function webcam_detection(~, ~)

    % Load CNN model
    if ~exist('trainedCNN.mat','file')
        msgbox('CNN model not found! Train using train_cnn.m first.', 'Error', 'error');
        return;
    end

    data = load('trainedCNN.mat');
    trainedCNN = data.trainedCNN;

    faceDetector = vision.CascadeObjectDetector;

    % ---------------------------
    %     MAIN RESTART LOOP
    % ---------------------------
    while ishandle(hFig)

        % Reset key so old keypress does not block S
        set(hFig,'CurrentCharacter',' ');

        % ---- WAIT FOR S TO START ----
        disp("Press S to start webcam");
        while true
            pause(0.1);
            if ~ishandle(hFig)
    return;
end
key = get(hFig,'CurrentCharacter');


            if strcmpi(key,'s')
                break; % start webcam
            elseif strcmpi(key,'q')
                return; % exit completely
            end
        end

        % ---- START WEBCAM ----
        try
            cam = webcam;
        catch
            msgbox('Webcam in use. Close other programs.','Error','error');
            return;
        end

        disp("Webcam Started");

        tic;
        frameCount = 0;

        % Reset key again before running
        set(hFig,'CurrentCharacter',' ');

        % -------- WEBCAM LOOP --------
        while ishandle(hFig)
            frame = snapshot(cam);
            bboxes = faceDetector(frame);

            for i = 1:size(bboxes,1)
                try
                    faceImg = imcrop(frame, bboxes(i,:));
                    faceImg = imresize(faceImg, [224 224]);

                    [label, score] = classify(trainedCNN, faceImg);
                    conf = max(score) * 100;

                    annotation = sprintf('%s (%.1f%%)', char(label), conf);

                    frame = insertObjectAnnotation(frame,'rectangle',...
                        bboxes(i,:), annotation,'LineWidth',3,'FontSize',30);

                catch
                    continue;
                end
            end

            % FPS
            frameCount = frameCount + 1;
            fps = frameCount / toc;

            frame = insertText(frame, [10 10], sprintf('FPS: %.1f',fps), ...
                'FontSize', 25, 'BoxOpacity', 0.6);

            axes(hAxes); 
            imshow(frame);
            title('Webcame Detection');
            drawnow;

            % Check for Q
            % ---- SAFETY CHECK (Prevents error when window is closed) ----
if ~ishandle(hFig)
    clear cam;
    return;  
end

key = get(hFig,'CurrentCharacter');
if strcmpi(key,'q')
    break;
end

        end

        % -------- CLEAN UP --------
        clear cam;
        set(hFig,'CurrentCharacter',' ');

        disp("Webcam Stopped. Press S again to Start.");

        % Loop restarts again => you can press S to start again
    end
end




    % -------------------------
    % DATABASE INFO
    % -------------------------
    function database_info(~,~)
        if ~exist('database','dir')
            msgbox('Database folder not found.','Error','error'); return;
        end
        imds = imageDatastore('database','IncludeSubfolders',true,'LabelSource','foldernames');
        tbl = countEachLabel(imds);
        msg = evalc('disp(tbl)');
        msgbox(msg, 'Database Info');
    end

    % -------------------------
    % REMOVE LABEL FOLDER
    % -------------------------
    function database_removal(~,~)
        answer = inputdlg('Enter Label Folder to Remove:','Remove Folder');
        if isempty(answer), return; end
        folder = fullfile('database',answer{1});
        if exist(folder,'dir')
            rmdir(folder,'s');
            msgbox(['Removed Folder: ',answer{1}],'Success');
        else
            msgbox('Folder not found.','Error','error');
        end
    end

    % -------------------------
    % PROGRAM INFO
    % -------------------------
    function program_info(~,~)
        info = sprintf(['Facial Expression Detection using FER-2013\n\n' ...
            'Features:\n' ...
            '✓ Image Detection\n' ...
            '✓ Real-Time Webcam Detection\n' ...
            '✓ CNN (VGG16)\n\n' ...
            'Toolboxes Required:\n' ...
            '• Deep Learning Toolbox\n' ...
            '• Image Processing Toolbox']);
        msgbox(info,'Program Info');
    end

    % -------------------------
    % EXIT
    % -------------------------
    function exit_program(~,~)
        choice = questdlg('Exit Application?','Exit','Yes','No','No');
        if strcmp(choice,'Yes'), close all; end
    end
end
