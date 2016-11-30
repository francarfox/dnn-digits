#library(ggplot2)
library(class)
library(MASS)

#FUNCTIONS
normalizePixels <- function(x) {
    return(x / 255)
}

applySVD <- function(x, dims=dims) {
    u <- svd(data.matrix(x)).u
    return (data.frame(u)[,0:dims])
}

generateVector <- function(number, length=nnumbers) {
    vector <- vector(mode="numeric", length=nnumbers)

    for (i in 1:length(vector)) {
        if (i == (number+1)) {
            vector[i] <- 1
        }
    }

    return (vector) 
}

generateOutput <- function(labels, numbers=nnumbers) {
    output <- matrix(0, 0, nnumbers)

    for (i in 1:length(labels)) {
        newout <- generateVector(labels[i], length=nnumbers)
        output <- rbind(output, newout)
    }

    return (output)
}

# Train: build and train a n-layers neural network
# input: is a data frame
dnn <- function(input=input, labels=labels, test=test, nhlayers=nhlayers, error=error) {
    print('Initialize train')

    delta <- 0
    nregisters <- nrow(input)
    diminput <- ncol(input)
    rate <- (error * nregisters) / 100

    # Initialize dimension for output of each layer
    dimoutput1 <- 100
    dimoutput2 <- 20
    dimoutputn <- ncol(labels)

    # Create the weight array
    w1 <- matrix(runif(diminput * dimoutput1, min=0, max=1), ncol=dimoutput1)
    w2 <- matrix(runif(ncol(w1) * dimoutput2, min=0, max=1), ncol=dimoutput2)
    w3 <- matrix(runif(ncol(w2) * dimoutputn, min=0, max=1), ncol=dimoutputn)


    # Number of runs
    runsCounter <- 0
    errors <- 100000

    while (errors > rate) {
        errors <- 0
        lastRegister <- 0

        # Init process for each register
        for (index in 1:nregisters) {
            inputVector <- input[index,];
            label <- labels[index]
            
            errors <- processLayers(input=inputVector, label=label, w1=w1, w2=w2, w3=w3, errors=errors)

            if (errors > rate) {
                lastRegister <- index
                break
            }
        }

        print(paste('runs:', runsCounter, 'lastRegister:', lastRegister))
        runsCounter <- runsCounter + 1
    }

    # Test
    prediction <- testDNN(dnn, test)
}

    # Calculate out deltas
    # for (index in nlayers:1) {
    #     if (index == nlayers) {
    #         # Compare to the label values
    #         outputDelta <- layerOutput[index] - labels
    #         error <- 
    #     } else {
    #         # Compare to the following layer's delta
    #         deltaPullback <- weights[index] * delta[-1]
    #         delta.append(deltaPullback[,] * sigmoid(layerInput))
    #     }
    # }

    # Compute weight deltas
    # for (index in 1:nlayers) {
    #     deltaIndex <- nlayers - index

    #     if (index == 0) {
    #         layerOutput <- 
    #     } else {
    #         layerOutput <- 
    #     }
    # }

processLayers <- function(input=input, label=label, w1=w1, w2=w2, w3=w3, errors=errors) {
    currentErrors <- errors

    # Process input leyer
    # input is a register no a matrix
    output1 <- processInputLayer(input=input, weights=w1)

    # Process hidden layers
    #for (index in 1:nhlayers) {
    output2 <- processHiddenLayer(input=output1, weights=w2)
    #}

    # Process output layer
    output <- processOutputLayer(input=output2, weights=w3)

    # Verify if is correct the output
    if (output != label) {
        # Recalculate weights
        x1 <- data.matrix(output1)
        x2 <- data.matrix(output2)
        x3 <- data.matrix(output)
        y <- 0.5

        w1 <- calculateWeights(w1, x=x1, y=y)
        w2 <- calculateWeights(w2, x=x2, y=y)
        w3 <- calculateWeights(w3, x=x3, y=y)

        currentErrors <- currentErrors + 1
    }

    return (currentErrors)
}

# input: a input register
# weights: weights of current layer. is a matrix 
processInputLayer <- function(input=input, weights=weights) {
    print('processInputLayer')

    # Initialize the network
    diminput <- ncol(input)
    dimoutput <- ncol(weights)
    summatory <- 0
    layerOutput <- c()
    
    # Generate all values for output neurons
    for (j in 1:dimoutput) {

        # Summatory
        for (i in 1:diminput) {
            delta <- weights[i,j] * input[i]
            summatory <- summatory + delta
        }

        # Apply activation function
        result <- sigmoid(summatory)
        layerOutput <- c(layerOutput, result)
    }

    return (data.frame(matrix(layerOutput, nrow=1)))
}

processHiddenLayer <- function(input=input, weights=weights) {
    print('processHiddenLayer')

    # Process the input with hidden layer
    diminput <- ncol(input)
    dimoutput <- ncol(weights)
    summatory <- 0
    layerOutput <- c()

    # Run it!
    for (j in 1:dimoutput) {

        # Summatory
        for (i in 1:diminput) {
            numbera <- as.numeric(weights[i,j])
            numberb <- as.numeric(input[1,i])

            delta <- numbera * numberb
            summatory <- summatory + delta
            #print(paste('delta:', delta, 'summatory:', summatory))
        }

        # Apply activation function
        result <- sigmoid(summatory)
        layerOutput <- c(layerOutput, result)
    }    

    return (data.frame(matrix(layerOutput, nrow=1)))
}

processOutputLayer <- function(input=input, weights=weights) {
    print('processOutputLayer')

    # Process the input with hidden layer
    diminput <- ncol(input)
    dimoutput <- ncol(weights)
    summatory <- 0
    layerOutput <- c()

    # Run it!
    for (j in 1:dimoutput) {

        # Summatory
        for (i in 1:diminput) {
            numbera <- as.numeric(weights[i,j])
            numberb <- as.numeric(input[1,i])

            delta <- numbera * numberb
            summatory <- summatory + delta
        }

        # Apply activation function
        result <- sigmoid(summatory)
        layerOutput <- c(layerOutput, result)
    }    

    return (data.frame(matrix(layerOutput, nrow=1)))
}

training <- function(input=input, labels=labels, rate=rate) {
    # Initialize the network
    nregisters <- nrow(input)
    dimensions <- ncol(input)
    nclasses <- ncol(labels)

    # Number of runs
    runsCounter <- 0
    errors <- 100000

    while (errors > rate) {
        errors <- 0
        lastRegister <- 0

        #
        for (index in 1:nregisters) {
            inputVector <- input[index,];
            summatory <- 0
            
            # Summatory
            for (i in 1:dimensions) {
                delta <- weights[i] * inputVector[i]
                summatory <- summatory + delta
            }

            # Apply activation function
            result <- round(sigmoid(summatory), digits=0)
            resultVector <- generateVector(result, length=nclasses)
            label <- getNumberFromVector(labels[index,])

            print(paste('index:', index, 'result:', result, 'label:', label))

            if (label != result) {
                x <- data.matrix(input[index,])
                y <- data.matrix(labels[index,])

                weights <- calculateWeights2(weights, x=x, y=y)
                errors <- errors + 1
            }

            if (errors > rate) {
                lastRegister <- index
                break
            }
        }

        print(paste('runs:', runsCounter, 'lastRegister:', lastRegister))
        runsCounter <- runsCounter + 1
    }
}

sigmoid <- function(x) {
    return (1 / 1 + exp(-x))
}

calculateWeights <- function(w, x=x, y=y) {
    result <- matrix(0, nrow=0, ncol=ncol(w))

    for (i in 1:nrow(w)) {
        wrow <- w[i,]
        delta <- wrow + x * y
        result <- rbind(result, delta)
    }

    print(paste('nrow result:', nrow(result), 'ncol result:', ncol(result)))
    return (result)
}

calculateWeights2 <- function(w, x=x, y=y) {
    number <- getNumberFromVector(y)
    return (w + x * number)
}

getNumberFromVector <- function(x) {
    number <- 0

    for (i in 1:length(x)) {
        if (x[i] == 1) {
            number <- i
        }
    }

    return (number - 1)
}


#MAIN - DNN Algorithm
ndims <- 225 # 15*15
nnumbers <- 10
nhlayers <- 3
error <- 0.23

train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Normalize pixels
trainData <- normalizePixels(train[,-1])

# Reduce dimensions with SVD
trainReduced <- applySVD(trainData, dims=ndims)

# Prepare output for DNN
trainLabels <- generateOutput(train[,1], numbers=nnumbers)

# DNN Training - Prediction
prediction <- dnn(input=trainReduced, labels=trainLabels, test=test, nhlayers=nhlayers, error=error)

# Submission
submission <- data.frame(imageId=1:nrow(test), label=prediction)
write.csv(submission, "dnnsubmit.csv", row.names=FALSE) 

