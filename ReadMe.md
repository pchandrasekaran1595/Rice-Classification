Machine and Deep Learning Analysis on predicting the variety of rice based on grain information.

---

&nbsp;

### **CLI Arguments**
<pre>
Common
1. --ml - Flag that controls entry to perform machine learning analysis
2. --dl - Flag that controls entry to perform deep learning analysis


Machine Learning Specific
1. --model-name - Name of the Model to be used (Supported: lgr, gnb, knc, dtc, rfc, xgc)
2. --test       - Flag that controls entry into test mode (Not Implemented as yet) 


Deep Learning Specific
1. --bs        - Batch Size (Default: 256)
2. --lr        - Learning Rate (Default: 1e-3)
3. --wd        - Weight Decay (Default: 0)
4. --dp        - Dropout Amount
5. --scheduler - Needs two arguments; patience and eps
6. --epochs    - Number of training epochs (Default: 10)
7. --early     - Early stopping patience (Default: 5)
8. --hl        - Hidden Layers (Number of layers followed by number of neurons in each layer)
9. --test      - Flag that controls entry into test mode (Not Implemented as yet)     
</pre>

&nbsp;

---