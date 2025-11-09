# CIFAR10_Weight_Initialization_Study
PyTorch project analyzing how weight initialization impacts CNN performance on CIFAR-10.
🧠 CIFAR-10 Weight Initialization Experiment (Deep CNN)
🎯 Objective

To study the impact of different weight initialization techniques on the training performance, convergence, and final accuracy of a Deep Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset.

🧩 Initializations Compared
Initialization	        Description	                        Suitable For	           Expected Effect
Zero	                All weights = 0         	               None	             No learning (symmetry)
Normal	            Random from Normal(0,1)	               Basic networks	       Unstable gradients
Xavier (Glorot)	     Scales weights by fan-in/out	          Tanh/Sigmoid	        Balanced gradients
Kaiming (He)	       Scales by fan-in only	                  ReLU	             Stable gradients, fast learning
Orthogonal	        Orthogonal matrix initialization	        Deep CNNs	           Preserves variance
⚙️ Experimental Setup

Dataset: CIFAR-10

Architecture: Deep CNN

Framework: PyTorch

Optimizer: SGD

Loss Function: CrossEntropyLoss

Epochs: 10

Device: CPU

📈 Results Overview
Initialization	Final Test Accuracy (%)	Observation
Zero	10.00	No learning
Normal	17.64	Basic improvement
Xavier	19.24	Stable learning
Kaiming	20.46	Fastest convergence
Orthogonal	20.31	Smooth learning
🖼️ Visual Results
🔹 Accuracy Progression Heatmap

Shows how accuracy improves across epochs for each initialization method.

🔹 Final Test Accuracy Bar Chart

Compares final performance after 10 epochs.

🔹 Loss Curve

Visualizes how training and test loss evolve for each initialization.

🔬 Key Insights

❌ Zero Initialization → No learning (weights stuck due to symmetry).

⚖️ Normal Initialization → Works but unstable.

⚙️ Xavier Initialization → Good for tanh/sigmoid.

⚡ Kaiming Initialization → Best for ReLU-based CNNs.

🧩 Orthogonal Initialization → Very stable for deep models.

🧠 Learnings

Proper weight initialization:

Ensures stable gradient flow

Speeds up convergence

Prevents vanishing/exploding gradients

Improves final accuracy

🧰 Code Structure
├── 01_weight_initialization_code.py
├── init_results.json
├── model_<init>.pth
├── c4a92b7b-cffc-4bc1-a700-610a997e06b5.png  (heatmap)
├── ebeb5c81-5519-4f45-af5f-8f45898ae0b3.png  (bar chart)
├── b821a742-554c-46a6-b47e-5657cf9b781d.png  (loss curve)
└── README.md

👩‍🔬 Author

Manisha Kalekar
M.Sc. Inorganic Chemistry | Machine Learning & Deep Learning Enthusiast
Exploring AI for Drug Discovery & Chemical Modeling.

⭐ Citation / LinkedIn Feature Line

“Explored how different weight initialization methods affect CNN learning on CIFAR-10. Demonstrated Kaiming and Orthogonal as most effective for ReLU activations.”

🧾 Future Work

Extend to deeper CNNs or ResNets

Measure gradient norms layer-wise

Study the effect on training time and convergence rate

📚 Reference

He et al., 2015 – Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

Glorot & Bengio, 2010 – Understanding the difficulty of training deep feedforward neural networks
