// ============================================================
// NEURO LAB - SIMPLE, STABLE CALLBACKS
// Scilab 2026.1.0
// ============================================================

global NEUROLAB_HANDLES;
global NEUROLAB_MODEL;


// ============================================================
// DATASET
// ============================================================
function [X,Y,name] = nl_get_dataset(choice)
    X = [0 0; 0 1; 1 0; 1 1];

    if choice == 1 then
        Y = [0; 1; 1; 0];
        name = "XOR";
    elseif choice == 2 then
        Y = [0; 0; 0; 1];
        name = "AND";
    else
        Y = [0; 1; 1; 1];
        name = "OR";
    end
endfunction


// ============================================================
// ACTIVATION
// ============================================================
function A = nl_sigmoid(Z)
    A = 1 ./ (1 + exp(-Z));
endfunction


// ============================================================
// FORWARD PASS
// ============================================================
function [A1,A2] = nl_forward(X,W1,b1,W2,b2)
    Z1 = X * W1 + ones(size(X,1),1) * b1;
    A1 = nl_sigmoid(Z1);

    Z2 = A1 * W2 + ones(size(A1,1),1) * b2;
    A2 = nl_sigmoid(Z2);
endfunction


// ============================================================
// LOSS
// ============================================================
function loss = nl_loss(Y,A2)
    epsv = 1e-10;
    P = A2;

    low = find(P < epsv);
    high = find(P > 1-epsv);

    if size(low,"*") > 0 then
        P(low) = epsv;
    end

    if size(high,"*") > 0 then
        P(high) = 1-epsv;
    end

    loss = -mean(Y .* log(P) + (1-Y) .* log(1-P));
endfunction


// ============================================================
// CLEAR AXES
// ============================================================
function nl_clear_axes(ax)
    if size(ax.children,"*") > 0 then
        delete(ax.children);
    end
endfunction


// ============================================================
// MAIN TRAINING CALLBACK
// ============================================================
function train_from_gui()
    global NEUROLAB_HANDLES;
    global NEUROLAB_MODEL;

    dataset_value = get(NEUROLAB_HANDLES.dataset_menu,"value");
    learning_rate = evstr(get(NEUROLAB_HANDLES.learning_rate_input,"string"));
    epochs = round(evstr(get(NEUROLAB_HANDLES.epochs_input,"string")));
    hidden_size = round(evstr(get(NEUROLAB_HANDLES.hidden_input,"string")));

    if learning_rate <= 0 | learning_rate > 5 then
        set(NEUROLAB_HANDLES.status_text,"string","STATUS: Learning rate must be 0-5");
        return;
    end

    if epochs < 10 then
        set(NEUROLAB_HANDLES.status_text,"string","STATUS: Use at least 10 epochs");
        return;
    end

    if hidden_size < 1 then
        set(NEUROLAB_HANDLES.status_text,"string","STATUS: Hidden neurons must be >= 1");
        return;
    end

    if hidden_size > 12 then
        hidden_size = 12;
        set(NEUROLAB_HANDLES.hidden_input,"string","12");
    end

    [X,Y,dataset_name] = nl_get_dataset(dataset_value);

    set(NEUROLAB_HANDLES.status_text,"string","STATUS: TRAINING...");
    set(NEUROLAB_HANDLES.accuracy_text,"string","Accuracy: calculating...");
    set(NEUROLAB_HANDLES.loss_text,"string","Final Loss: calculating...");
    set(NEUROLAB_HANDLES.epoch_text,"string","Epochs: " + string(epochs));
    drawnow();

    // --------------------------------------------------------
    // INITIALIZE NETWORK
    // --------------------------------------------------------
    rand("seed",12345);

    W1 = (rand(2,hidden_size) - 0.5) * 2;
    b1 = zeros(1,hidden_size);

    W2 = (rand(hidden_size,1) - 0.5) * 2;
    b2 = 0;

    loss_history = zeros(epochs,1);
    m = size(X,1);

    // --------------------------------------------------------
    // TRAIN
    // --------------------------------------------------------
    for epoch = 1:epochs

        [A1,A2] = nl_forward(X,W1,b1,W2,b2);

        loss_history(epoch) = nl_loss(Y,A2);

        dZ2 = A2 - Y;
        dW2 = (A1' * dZ2) / m;
        db2 = sum(dZ2) / m;

        dA1 = dZ2 * W2';
        dZ1 = dA1 .* (A1 .* (1-A1));

        dW1 = (X' * dZ1) / m;
        db1 = sum(dZ1,"r") / m;

        W2 = W2 - learning_rate * dW2;
        b2 = b2 - learning_rate * db2;
        W1 = W1 - learning_rate * dW1;
        b1 = b1 - learning_rate * db1;
    end

    // --------------------------------------------------------
    // FINAL PREDICTION
    // --------------------------------------------------------
    [A1,A2] = nl_forward(X,W1,b1,W2,b2);

    predictions = bool2s(A2 >= 0.5);
    correct = sum(predictions == Y);
    accuracy = correct / size(Y,1) * 100;
    final_loss = loss_history($);

    // --------------------------------------------------------
    // STORE MODEL
    // --------------------------------------------------------
    NEUROLAB_MODEL = struct();

    NEUROLAB_MODEL.W1 = W1;
    NEUROLAB_MODEL.b1 = b1;
    NEUROLAB_MODEL.W2 = W2;
    NEUROLAB_MODEL.b2 = b2;

    NEUROLAB_MODEL.A1 = A1;
    NEUROLAB_MODEL.A2 = A2;

    NEUROLAB_MODEL.X = X;
    NEUROLAB_MODEL.Y = Y;
    NEUROLAB_MODEL.predictions = predictions;

    NEUROLAB_MODEL.loss_history = loss_history;
    NEUROLAB_MODEL.accuracy = accuracy;

    NEUROLAB_MODEL.dataset_name = dataset_name;
    NEUROLAB_MODEL.hidden_size = hidden_size;
    NEUROLAB_MODEL.learning_rate = learning_rate;
    NEUROLAB_MODEL.epochs = epochs;

    // --------------------------------------------------------
    // RESULT TEXT
    // --------------------------------------------------------
    set(NEUROLAB_HANDLES.status_text,"string","STATUS: TRAINING COMPLETE");
    set(NEUROLAB_HANDLES.accuracy_text,"string","Accuracy: " + string(round(accuracy)) + "%");
    set(NEUROLAB_HANDLES.loss_text,"string","Final Loss: " + string(round(final_loss*10000)/10000));
    set(NEUROLAB_HANDLES.epoch_text,"string","Epochs: " + string(epochs));

    // --------------------------------------------------------
    // PREDICTION TABLE
    // --------------------------------------------------------
    table_data = [ ...
        "X1" "X2" "Actual" "Predicted" "Probability"; ...
        string(X(1,1)) string(X(1,2)) string(Y(1)) string(predictions(1)) string(round(A2(1)*100)/100); ...
        string(X(2,1)) string(X(2,2)) string(Y(2)) string(predictions(2)) string(round(A2(2)*100)/100); ...
        string(X(3,1)) string(X(3,2)) string(Y(3)) string(predictions(3)) string(round(A2(3)*100)/100); ...
        string(X(4,1)) string(X(4,2)) string(Y(4)) string(predictions(4)) string(round(A2(4)*100)/100) ...
    ];

    set(NEUROLAB_HANDLES.prediction_table,"string",table_data);

    // --------------------------------------------------------
    // LEARNING INSIGHT
    // --------------------------------------------------------
    insight = ...
        "Dataset: " + dataset_name + ...
        " | Hidden neurons: " + string(hidden_size) + ...
        " | Learning rate: " + string(learning_rate) + ...
        ascii(10) + ...
        "Loss graph: binary cross-entropy versus epoch." + ...
        ascii(10) + ...
        "Network: CYAN = positive weight, RED = negative weight." + ...
        ascii(10) + ...
        "Input -> Hidden -> Output shows the learned architecture.";

    set(NEUROLAB_HANDLES.insight_text,"string",insight);

    // --------------------------------------------------------
    // DRAW GRAPH + NETWORK
    // --------------------------------------------------------
    draw_loss_graph(loss_history,epochs,dataset_name);
    draw_network_graph(W1,W2,A1,A2,hidden_size);

    drawnow();
endfunction


// ============================================================
// LOSS GRAPH
// ============================================================
function draw_loss_graph(loss_history,epochs,dataset_name)
    global NEUROLAB_HANDLES;

    scf(NEUROLAB_HANDLES.figure);
    sca(NEUROLAB_HANDLES.loss_axes);

    nl_clear_axes(NEUROLAB_HANDLES.loss_axes);

    h = plot(NEUROLAB_HANDLES.loss_axes,1:epochs,loss_history,"c-");

    if typeof(h) == "handle" then
        h.thickness = 3;
    end

    a = gca();

    a.background = [0.04 0.05 0.08];
    a.foreground = color("white");
    a.labels_font_color = color("white");
    a.font_color = color("white");
    a.font_size = 2;

    a.axes_visible = ["on" "on"];
    a.box = "on";
    a.auto_scale = "off";

    max_loss = max(loss_history);

    if max_loss <= 0 then
        max_loss = 1;
    end

    a.data_bounds = [1 0; epochs max_loss*1.10];
    a.tight_limits = "off";

    xtitle("Training Loss - " + dataset_name,"Epoch","Loss");
    xgrid();

    drawnow();
endfunction


// ============================================================
// NETWORK GRAPH
// ============================================================
function draw_network_graph(W1,W2,A1,A2,hidden_size)
    global NEUROLAB_HANDLES;

    scf(NEUROLAB_HANDLES.figure);
    sca(NEUROLAB_HANDLES.network_axes);

    nl_clear_axes(NEUROLAB_HANDLES.network_axes);

    // --------------------------------------------------------
    // POSITIONS
    // --------------------------------------------------------
    input_x = 1;
    hidden_x = 3;
    output_x = 5;

    input_y = [2.7 0.9];
    output_y = 1.8;

    if hidden_size == 1 then
        hidden_y = 1.8;
    else
        hidden_y = linspace(3.2,0.4,hidden_size);
    end

    // --------------------------------------------------------
    // INPUT -> HIDDEN
    // --------------------------------------------------------
    for i = 1:2
        for j = 1:hidden_size

            if W1(i,j) >= 0 then
                line_style = "c-";
            else
                line_style = "r-";
            end

            h = plot( ...
                NEUROLAB_HANDLES.network_axes, ...
                [input_x hidden_x], ...
                [input_y(i) hidden_y(j)], ...
                line_style);

            if typeof(h) == "handle" then
                h.thickness = 2 + min(3,abs(W1(i,j)));
            end
        end
    end

    // --------------------------------------------------------
    // HIDDEN -> OUTPUT
    // --------------------------------------------------------
    for j = 1:hidden_size

        if W2(j) >= 0 then
            line_style = "c-";
        else
            line_style = "r-";
        end

        h = plot( ...
            NEUROLAB_HANDLES.network_axes, ...
            [hidden_x output_x], ...
            [hidden_y(j) output_y], ...
            line_style);

        if typeof(h) == "handle" then
            h.thickness = 2 + min(3,abs(W2(j)));
        end
    end

    // --------------------------------------------------------
    // LARGE VISIBLE NODES
    // --------------------------------------------------------
    plot( ...
        NEUROLAB_HANDLES.network_axes, ...
        [input_x input_x], ...
        [input_y(1) input_y(2)], ...
        "co","markersize",14,"markerface","cyan");

    plot( ...
        NEUROLAB_HANDLES.network_axes, ...
        hidden_x*ones(1,hidden_size), ...
        hidden_y, ...
        "mo","markersize",14,"markerface","magenta");

    plot( ...
        NEUROLAB_HANDLES.network_axes, ...
        output_x,output_y, ...
        "go","markersize",16,"markerface","green");

    // --------------------------------------------------------
    // LABELS
    // --------------------------------------------------------
    xstring(0.65,3.55,"INPUT");
    xstring(2.55,3.55,"HIDDEN");
    xstring(4.65,3.55,"OUTPUT");

    xstring(0.45,input_y(1),"X1");
    xstring(0.45,input_y(2),"X2");
    xstring(5.15,output_y,"Y");

    for j = 1:hidden_size
        xstring(hidden_x+0.10,hidden_y(j),"H"+string(j));
    end

    // --------------------------------------------------------
    // AXES
    // --------------------------------------------------------
    a = gca();

    a.background = [0.04 0.05 0.08];
    a.foreground = color("white");
    a.labels_font_color = color("white");
    a.font_color = color("white");
    a.font_size = 2;

    a.axes_visible = ["on" "on"];
    a.box = "on";
    a.auto_scale = "off";

    a.data_bounds = [0 0; 6 4];
    a.tight_limits = "off";

    xtitle("Neural Network Flow","Layer","Neuron position");

    drawnow();
endfunction


// ============================================================
// RESET
// ============================================================
function reset_gui()
    global NEUROLAB_HANDLES;
    global NEUROLAB_MODEL;

    set(NEUROLAB_HANDLES.dataset_menu,"value",1);
    set(NEUROLAB_HANDLES.learning_rate_input,"string","0.5");
    set(NEUROLAB_HANDLES.epochs_input,"string","3000");
    set(NEUROLAB_HANDLES.hidden_input,"string","4");

    set(NEUROLAB_HANDLES.status_text,"string","STATUS: READY");
    set(NEUROLAB_HANDLES.accuracy_text,"string","Accuracy: --");
    set(NEUROLAB_HANDLES.loss_text,"string","Final Loss: --");
    set(NEUROLAB_HANDLES.epoch_text,"string","Epochs: 0");

    empty_table = [ ...
        "X1" "X2" "Actual" "Predicted" "Probability"; ...
        "-" "-" "-" "-" "-"; ...
        "-" "-" "-" "-" "-"; ...
        "-" "-" "-" "-" "-"; ...
        "-" "-" "-" "-" "-" ...
    ];

    set(NEUROLAB_HANDLES.prediction_table,"string",empty_table);

    set(NEUROLAB_HANDLES.insight_text, ...
        "string","Choose XOR, AND or OR and press TRAIN NETWORK.");

    scf(NEUROLAB_HANDLES.figure);

    sca(NEUROLAB_HANDLES.loss_axes);
    nl_clear_axes(NEUROLAB_HANDLES.loss_axes);
    xtitle("Training Loss","Epoch","Loss");
    xgrid();

    a = gca();
    a.background = [0.04 0.05 0.08];
    a.foreground = color("white");
    a.labels_font_color = color("white");
    a.font_color = color("white");

    sca(NEUROLAB_HANDLES.network_axes);
    nl_clear_axes(NEUROLAB_HANDLES.network_axes);
    xtitle("Neural Network Flow","Layer","Neuron position");

    a = gca();
    a.background = [0.04 0.05 0.08];
    a.foreground = color("white");
    a.labels_font_color = color("white");
    a.font_color = color("white");
    a.data_bounds = [0 0; 6 4];

    NEUROLAB_MODEL = struct();

    drawnow();
endfunction


// ============================================================
// SHOW MATHEMATICS
// ============================================================
function show_math()
    global NEUROLAB_HANDLES;

    set(NEUROLAB_HANDLES.insight_text, ...
        "string", ...
        "Forward: Z1 = X*W1 + b1" + ascii(10) + ...
        "Hidden: A1 = sigmoid(Z1)" + ascii(10) + ...
        "Output: A2 = sigmoid(A1*W2 + b2)" + ascii(10) + ...
        "Loss: Binary Cross-Entropy" + ascii(10) + ...
        "Update: W = W - learning_rate * gradient");
endfunction


// ============================================================
// INSPECT NETWORK
// ============================================================
function inspect_network()
    global NEUROLAB_HANDLES;
    global NEUROLAB_MODEL;

    if ~isfield(NEUROLAB_MODEL,"W1") then
        set(NEUROLAB_HANDLES.insight_text, ...
            "string","No trained network yet. Press TRAIN NETWORK first.");
        return;
    end

    W1 = NEUROLAB_MODEL.W1;
    W2 = NEUROLAB_MODEL.W2;
    A2 = NEUROLAB_MODEL.A2;

    txt = ...
        "NETWORK INSPECTION" + ascii(10) + ...
        "Input layer: 2 neurons" + ascii(10) + ...
        "Hidden layer: " + string(size(W1,2)) + " neurons" + ascii(10) + ...
        "Output layer: 1 neuron" + ascii(10) + ...
        "W1 size: " + string(size(W1,1)) + " x " + string(size(W1,2)) + ascii(10) + ...
        "W2 size: " + string(size(W2,1)) + " x 1" + ascii(10) + ...
        "Sample 1 probability: " + string(round(A2(1)*100)/100);

    set(NEUROLAB_HANDLES.insight_text,"string",txt);
endfunction
