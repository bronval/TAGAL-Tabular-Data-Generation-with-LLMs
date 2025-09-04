#################################################################################################
#
# Implements the class used to serialize the data
#
#################################################################################################

from dataloader import Dataloader
import pandas as pd


class Serializer:

    def serialize_examples(self, examples: pd.DataFrame = None, examples_range: int|range = None, ignore_cols: list[str] = []) -> str:
        raise NotImplementedError
    
    def deserialize_examples(self, examples_str: str, verbose: bool = True) -> pd.DataFrame:
        raise NotImplementedError



class SentenceSerializer:

    def __init__(self, dataloader: Dataloader, specifier: str, value_sep: str, feature_sep: str, end_of_line: str):
        """
        Creates a serializer object with the given elements. The serializer is different for each dataset.
        The format for the serialization is: 'specifier [feature name] value_sep [feature value] feature_sep' (without spaces)

        Input:
        - dataloader: dataloader with the data to serialize
        - specifier: string, the specifier at the start of the sentence
        - value_sep: string, the separator between the feature name and the value
        - feature_sep: string, the separator between the features
        - end_of_line: string, the end of line character to use between the examples
        """
        self.serializer_type = "sentence"
        self.specifier = specifier
        self.value_sep = value_sep
        self.feature_sep = feature_sep
        self.end_of_line = end_of_line
        self.dataloader = dataloader
        self.data = dataloader.data
        self.features = self.data.columns
        self.cat_features = dataloader.cat_features
        self.features_order_type = "original"

        # build set of values for unique values of categorical features, used to ensure generated examples are correct
        self.cat_values = {}
        for feat in self.cat_features:
            self.cat_values[feat] = set(self.data[feat].unique())
        if dataloader.task == "classification":
            self.cat_values[dataloader.target] = set(self.data[dataloader.target].unique())

        self.dtype_map = {feat: type(self.data[feat][0]) for feat in self.features}

    
    def __serialize_example(self, example: pd.Series) -> str:
        """
        Serializes an example and returns the result as a string

        Input:
        - example: pandas Series, the example to serialize
        """
        out = ""
        for feat, val in example.items():
            out += f"{self.specifier}{feat}{self.value_sep}{val}{self.feature_sep}"
        out = out.strip(self.feature_sep)
        return out

    
    def serialize_examples(self, examples: pd.DataFrame = None, examples_range: int|range = None) -> str:
        """
        Serializes the examples given in a DataFrame and returns the result as a string

        Input:
        - examples: DataFrame, the examples to serialize. If None, consider the examples_range parameter. Default=None
        - examples_range: int or range, the range of examples to serialize. Used only if examples parameter is None. If examples_range=None, serializes all examples, if int serializes the range (0, examples_range), otherwise serializes the range. Default=None
        """
        out = ""
        if examples is None:
            examples = self.data
            if examples_range is not None:
                if isinstance(examples_range, int):
                    examples = examples.iloc[:examples_range]
                elif isinstance(examples_range, range):
                    examples = examples.iloc[examples_range]
                else:
                    raise ValueError("examples_range must be an int or a range")

        for _, example in examples.iterrows():
            out += self.__serialize_example(example) + self.end_of_line
        out = out.strip(self.end_of_line)
        return out
    

    def __check_deserialized(self, example: dict) -> bool:
        """
        Checks if the deserialized example is valid. Returns True if it is, False otherwise.

        Input:
        - example: dictionary, the deserialized example
        """
        if example is None or len(example) == 0 or len(example) != len(self.features):
            return False
        for feat in self.features:
            if feat not in example:
                return False
        cat_feats = [feat for feat in self.cat_features]
        if self.dataloader.task == "classification":
            cat_feats.append(self.dataloader.target)
        for feat in cat_feats:
            if example[feat] not in self.cat_values[feat]:
                return False
        return True

    
    def __deserialize_example(self, example_str: str) -> dict:
        """
        From a serialized example string, returns a dictionary with the feature names and values

        Input:
        - example_str: string, the serialized example
        """
        example_feats = example_str.split(self.feature_sep)
        example = {}
        for feat in example_feats:
            try:
                feat = feat.strip().split(self.specifier)[1]
            except IndexError:
                return None
            if feat:
                try:
                    feat_name, feat_val = feat.split(self.value_sep)
                    example[feat_name] = self.dtype_map[feat_name](feat_val)
                except (ValueError, KeyError):
                    return None

        if not self.__check_deserialized(example):
            return None
        return example


    def deserialize_examples(self, examples_str: str, verbose: bool = True) -> pd.DataFrame:
        """
        From a text examples_str with the serialized examples, returns a DataFrame with the deserialized examples. Returns None if no valid examples were extracted.

        Input:
        - examples_str: string, the serialized examples with the same format as the one used to serialize the examples
        """
        examples = examples_str.split(self.end_of_line)
        examples = [self.__deserialize_example(example) for example in examples]
        examples_valid = [example for example in examples if example is not None]
        diff = len(examples) - len(examples_valid)
        if verbose and diff > 0:
            print(f"Warning: {diff} examples were not valid and skipped during deserialization")
        if len(examples_valid) == 0:
            if verbose:
                print("Warning: no valid examples were obtained during deserialization. Returning None.")
            return None
        return pd.DataFrame(examples_valid)


    def get_format(self) -> str:
        """
        Returns a string with the format used to serialize the examples
        """
        return f"{self.specifier}[feature name]{self.value_sep}[feature value]{self.feature_sep}"


    def __str__(self):
        s = f"Serializer object for dataset {self.dataloader.dname}\n"
        s += f"Specifier: '{self.specifier}'\n"
        s += f"Value separator: '{self.value_sep}'\n"
        s += f"Feature separator: '{self.feature_sep}'\n"
        s += f"Serialized format: {self.specifier}[feature name]{self.value_sep}[feature value]{self.feature_sep}"
        return s



class CSVSerializer:

    def __init__(self, dataloader: Dataloader, feat_order: str = "original"):
        """
        Creates a CSVSerializer object with the given dataloader

        Input:
        - dataloader: dataloader with the data to serialize
        - feat_order: string, in which order should the features be serialized. 'original' for order from the dataset, 'num_first', or 'cat_first' for numerical/categorical features first. The target will always be at the end. Default='original'
        """
        self.serializer_type = "csv"
        self.dataloader = dataloader
        self.data = dataloader.data
        self.features = self.data.columns
        self.cat_features = dataloader.cat_features
        
        if feat_order not in ["original", "num_first", "cat_first"]:
            raise ValueError(f"feat_order must be one of 'original', 'num_first', 'cat_first', got {feat_order}")
        self.features_order_type = feat_order

        # build set of values for unique values of categorical features, used to ensure generated examples are correct
        self.cat_values = {}
        for feat in self.cat_features:
            self.cat_values[feat] = set(self.data[feat].unique())
        if dataloader.task == "classification":
            self.cat_values[dataloader.target] = set(self.data[dataloader.target].unique())

        self.dtype_map = {feat: type(self.data[feat][0]) for feat in self.features}

        self.features_order = []
        if self.features_order_type == "original":
            self.features_order = self.features.copy()
        elif self.features_order_type == "num_first":
            self.features_order = [feat for feat in self.features if feat not in self.cat_features and feat != self.dataloader.target]
            self.features_order += [feat for feat in self.cat_features if feat != self.dataloader.target]
            self.features_order += [self.dataloader.target]
        else:
            self.features_order = [feat for feat in self.cat_features if feat != self.dataloader.target]
            self.features_order += [feat for feat in self.features if feat not in self.cat_features and feat != self.dataloader.target]
            self.features_order += [self.dataloader.target]



    def serialize_examples(self, examples: pd.DataFrame = None, examples_range: int|range = None) -> str:
        out = ""
        # first add the names of the features
        out = ",".join([feat for feat in self.features_order]) + "\n"

        # serialize the examples in a csv format
        if examples is None:
            examples = self.data
            if examples_range is not None:
                if isinstance(examples_range, int):
                    examples = examples.iloc[:examples_range]
                elif isinstance(examples_range, range):
                    examples = examples.iloc[examples_range]
                else:
                    raise ValueError("examples_range must be an int or a range")
                
        for _, example in examples.iterrows():
            # out += ",".join([str(val) for feat, val in example.items()]) + "\n"
            out += ",".join([str(example[feat]) for feat in self.features_order]) + "\n"
        out = out.strip("\n")
        
        return out


    def __check_deserialized(self, example: dict) -> bool:
        if example is None or len(example) == 0 or len(example) != len(self.features):
            return False
        # check that all the features are in the example
        for feat in self.features:
            if feat not in example:
                return False
        # check that the values of the categorical features are correct
        cat_feats = [feat for feat in self.cat_features]
        if self.dataloader.task == "classification":
            cat_feats.append(self.dataloader.target)
        for feat in cat_feats:
            if example[feat] not in self.cat_values[feat]:
                return False
        return True


    def __convert_line_to_dict(self, line: str) -> dict:
        line = line.split(",")
        if len(line) != len(self.features):
            return None
        # check that line is not just the names of the features
        if all([feat in self.features for feat in line]):
            return None
        example = {}
        for i, feat in enumerate(self.features_order):
            try:
                example[feat] = self.dtype_map[feat](line[i])
                if isinstance(example[feat], str):
                    example[feat] = example[feat].strip()
            except ValueError:
                return None
        if not self.__check_deserialized(example):
            return None
        return example


    def deserialize_examples(self, examples_str: str, verbose: bool = True) -> pd.DataFrame:
        examples = examples_str.split("\n")
        examples = [self.__convert_line_to_dict(example) for example in examples]
        valid_examples = [ex for ex in examples if ex is not None]
        if len(valid_examples) == 0:
            if verbose:
                print("Warning: no valid examples were obtained during deserialization. Returning None.")
            return None
        elif verbose and len(valid_examples) < len(examples):
            print(f"Warning: {len(examples) - len(valid_examples)} examples were not valid and skipped during deserialization")
        df = pd.DataFrame(valid_examples)
        return df


