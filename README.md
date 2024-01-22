# KGCE
This repository is the implementation of KGCE

---
## Pre-requisits

Running environment
*   Python  3.8
*   Cuda  11.8
*   PyTorch  2.0.0
*   numpy 1.24.2
*   scikit-learn 1.3.2




## Files in the folder

- `data/`
  - `dblp/`
    - `dblp.csv`: raw rating file of BDLP dataset;
    - `data_dblp.pkl`: initial embeddings and adjacency matrices of KG;
    - ...
  
  - `yelp/`
    - `yelp.csv`: raw rating file of Yelp dataset;
    - `data_yelp.pkl`: initial embeddings and adjacency matrices of KG;
    - ...
  - `city/`
    - `city.csv`: raw rating file of Foursquare dataset;
    - `data_city.pkl`: initial embeddings and adjacency matrices of KG;
    - ...
- `gene/`: sampling path of data.
- `src/`: implementations of KGCE.


## Data description

The data in the pkl file is in the following format：

```bash
#dict
data = {} 

#the initial embedding of nodes, the key is node type.
data['feature'] = {'P':p_emb, 'A':a_emb,'V':v_emb,'C':c_emb} 

#Hierarchical tree structures(HTS), i.e., VPA, APV.
data['HTS']=[['P','A'],['A','P','V','C']]

#The adjacency matrix between each two levels in each hierarchical tree
data['adjs']=[[AP,PV],[PA,VP,CV,VC]]
```

## Running

* dblp

    ```python
    !python src/main.py --dataset dblp --lambda1 0.1 --epoch 30 --seed 21
    ```
* yelp
    ```python
    !python src/main.py --dataset yelp --lambda1 1 --epoch 20 --seed 21 
    ```

* foursquare
    ```python
    !python src/main.py --dataset city --lr_emb 0.001 --lr_rs 0.001 --l2 0.0001  --lambda1 100  --seed 21
    ```


## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please create a new issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).


