## The Backbone

We are using ResNet-50, pre-trained on the standard ImageNet dataset. As per the project requirements, both predictions must be produced by a shared backbone.  

However, a standard ResNet-50 is designed to classify 1,000 different types of objects. To make it work for our specific forensic tasks, we perform a crucial "amputation": we strip off its final classification layer (the fc layer in PyTorch) and replace it with an empty placeholder (nn.Identity()).

Instead of outputting 1,000 class probabilities, the backbone now acts purely as a feature extractor. When an image passes through it, it outputs a raw, dense vector of 2,048 numbers representing the deepest patterns it found in the image.

## The Two Heads

Once we have that 2,048-dimensional feature vector from the shared backbone, we branch it off into two separate, independent paths. You are absolutely correct: these are just simple Linear (Fully Connected) layers tacked onto the end.

Head 1 (Real vs. Fake): A single nn.Linear layer that takes the 2,048 features and compresses them down to 1 single output number. This is your raw logit indicating if the image is an AI generation or a real photograph.

Head 2 (Transformation): Another nn.Linear layer sitting directly parallel to the first one. It takes the exact same 2,048 features and maps them to 3 output numbers. These are the raw logits for your three post-processing classes (original, internet-transmitted, or re-digitized).

During training, because both heads are bolted to the same backbone and are trained jointly, the ResNet-50 is forced to learn visual features that are universally useful for solving both tasks simultaneously, while the linear heads specialize in making the final specific decisions.

## The Neural Network (MultiTaskModel)

You built a Multi-Task Learning (MTL) architecture designed to solve two problems at the exact same time:

The Shared Backbone: You imported a pre-trained ResNet-50 and surgically removed its final classification layer. Now, instead of guessing what an image is, it acts as a powerful feature extractor, digesting a (224, 224) image into a rich 2,048-dimensional feature vector.

Head 1 (Real vs. Fake): You added a linear layer that compresses those 2,048 features into a single number (a logit) to predict if the image is an AI generation or not.

Head 2 (Transformation): You added a parallel linear layer that maps those same 2,048 features into 3 numbers to predict the post-processing history (Original, Transmitted, or Re-digitized).

## The Math & Optimization (MultiTaskLoss & Training Loop)
   
You engineered the logic that allows the model to actually learn. The Joint Loss Function:
You combined BCE (Binary Cross Entropy) With Logits Loss (for the binary Real/Fake task) and Cross Entropy Loss (for the 3-class transformation task). You added tunable weights ($\alpha$ and $\beta$) so you can control which task the model should prioritize during your ablation studies.The Training Loop: You wrote a robust train_epoch function that handles the complete deep learning lifecycle: moving data to the GPU, making predictions (forward pass), calculating the error, and using the Adam optimizer to update the network's weights (backward pass).