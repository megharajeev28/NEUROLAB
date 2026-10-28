// ============================================================
// NEURO LAB - SIMPLE NEURAL NETWORK CORE
// Scilab 2026 compatible
// ============================================================

function [X,Y,name] = get_dataset(choice)
    if choice == 1 then
        X = [0 0; 0 1; 1 0; 1 1];
        Y = [0; 1; 1; 0];
        name = "XOR";
    elseif choice == 2 then
        X = [0 0; 0 1; 1 0; 1 1];
        Y = [0; 0; 0; 1];
        name = "AND";
    else
        X = [0 0; 0 1; 1 0; 1 1];
        Y = [0; 1; 1; 1];
        name = "OR";
    end
endfunction

function A = sigmoid(Z)
    A = 1 ./ (1 + exp(-Z));
endfunction

function [W1,b1,W2,b2] = initialize_network(hidden_size)
    rand("seed", 12345);
    W1 = (rand(2,hidden_size) - 0.5) * 2;
    b1 = zeros(1,hidden_size);
    W2 = (rand(hidden_size,1) - 0.5) * 2;
    b2 = 0;
endfunction

function [A1,A2] = forward_network(X,W1,b1,W2,b2)
    Z1 = X * W1 + ones(size(X,1),1) * b1;
    A1 = sigmoid(Z1);
    Z2 = A1 * W2 + ones(size(A1,1),1) * b2;
    A2 = sigmoid(Z2);
endfunction

function loss = binary_loss(Y,A2)
    epsv = 1e-10;
    P = min(max(A2,epsv),1-epsv);
    loss = -mean(Y .* log(P) + (1-Y) .* log(1-P));
endfunction

function [W1,b1,W2,b2,loss_history] = train_network(X,Y,hidden_size,learning_rate,epochs)
    [W1,b1,W2,b2] = initialize_network(hidden_size);
    loss_history = zeros(epochs,1);
    m = size(X,1);

    for epoch = 1:epochs
        [A1,A2] = forward_network(X,W1,b1,W2,b2);

        loss_history(epoch) = binary_loss(Y,A2);

        // Output sigmoid + binary cross entropy derivative
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
endfunction
