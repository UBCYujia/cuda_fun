using CUDA

function sigmoid(x)
    return 1.0 ./ (1 .+ exp.(-x))
end

function sigmoid_derivative(x)
    return sigmoid(x) .* (1 .- sigmoid(x))
end


function forward_pass_kernel!(input, weights, biases, output, z, N)
    i = threadIdx().x
    if i <= N
        accumulator = 0.0f0
        for j in axes(weights,2)
            accumulator += weights[i, j] * input[j]
        end
        z[i] = accumulator + biases[i]
        output[i] = sigmoid(z[i])
    end
    return nothing
end


function backward_pass_kernel!(input, output, z, target, weights, biases, d_weights, d_biases, learning_rate, N)
    i = threadIdx().x
    if i <= N
        d_loss_output = 2.0 * (output[i] - target[i])
        d_output_z = sigmoid_derivative(z[i])
        d_loss_z = d_loss_output * d_output_z

        for j in axes(input, 1)
            d_weights[i, j] = d_loss_z * input[j]
        end

        d_biases[i] = d_loss_z
    end
    return nothing
end

function update_weights!(weights, d_weights, learning_rate)
    weights .-= learning_rate .* d_weights
end

function update_biases!(biases, d_biases, learning_rate)
    biases .-= learning_rate .* d_biases
end


CUDA.allowscalar(false)

input = CUDA.fill(0.5f0, 2, 1)
weights = CUDA.rand(2, 2)
biases = CUDA.rand(2, 1)
target = CUDA.fill(0.3f0, 2, 1)

d_weights = CUDA.zeros(2, 2)
d_biases = CUDA.zeros(2, 1)
output = CUDA.zeros(2, 1)
z = CUDA.zeros(2, 1)

learning_rate = 0.1
epochs = 1000
N = size(weights, 1)


for epoch in 1:epochs

    @cuda threads=N forward_pass_kernel!(input, weights, biases, output, z, N)
    
    @cuda threads=N backward_pass_kernel!(input, output, z, target, weights, biases, d_weights, d_biases, learning_rate, N)
    
    current_loss = mse_loss(output, target)
    
    update_weights!(weights, d_weights, learning_rate)
    update_biases!(biases, d_biases, learning_rate)
    
    if epoch % 100 == 0
        println("epoch $epoch, loss: $(CUDA.collect(current_loss))")  
    end
end


@cuda threads=N forward_pass_kernel!(input, weights, biases, output, z, N)
final_output = CUDA.collect(output)
println("final output: ", final_output)

