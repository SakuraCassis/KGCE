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


## Dataset Download

The dataset for the KGCE project is available for download from the following link:

[Download Dataset](insert_link_here)

Please download the dataset from the provided link and place it in the "data" folder.

## Running

* dblp

    ```python
    python src/main.py --dataset dblp --lambda1 0.1 --epoch 30 --seed 21
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

## Contact

For any questions or inquiries, please contact [1561659623@qq.com](mailto:1561659623@qq.com).
