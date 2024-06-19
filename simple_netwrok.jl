using CUDA

function sigmoid(x)
    return 1.0 ./ (1 .+ exp.(-x))
end

function sigmoid_derivative(x)
    return sigmoid(x) .* (1 .- sigmoid(x))
end

function forward_pass(input, weights, biases)
    z = weights * input .+ biases
    return sigmoid.(z), z
end

function mse_loss(output, target)
    return sum((output .- target).^2) / length(output)
end

function backward_pass(input, output, z, target, weights, biases, learning_rate)
    d_loss_output = 2.0 * (output - target)
    d_output_z = sigmoid_derivative(z)
    d_loss_z = d_loss_output .* d_output_z
    d_loss_weights = d_loss_z * input'
    d_loss_biases = d_loss_z
    #update weights and biases
    weights .-= learning_rate * d_loss_weights
    biases .-= learning_rate * d_loss_biases
end

CUDA.allowscalar(false)  

input = CUDA.fill(0.5f0, 2, 1)    
weights = CUDA.rand(2, 2)          
biases = CUDA.rand(2, 1)           
target = CuArray{Float32}([0.2,0.4])    

learning_rate = 0.1
epochs = 1000

for epoch in 1:epochs
    output, z = forward_pass(input, weights, biases)
    loss = mse_loss(output, target)
    backward_pass(input, output, z, target, weights, biases, learning_rate)
    
    if epoch % 100 == 0
        println("Epoch $epoch, Loss: $loss")
    end
end

final_output = forward_pass(input, weights, biases)[1]
println("Final output: ", Array(final_output))

