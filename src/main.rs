use csv::{Reader, Writer};
use std::{fs::File, io::Write};
use ndarray::{ Array, Array1, Array2 };
use linfa::Dataset;
use linfa_trees::{DecisionTree, SplitQuality};
use linfa::prelude::*;

fn main() {
  let dataset = get_dataset();
  println!("{:?}", dataset);
  let (train, test) = dataset.clone().split_with_ratio(0.9); //splits the dataset with 90% of it used to train the model and 10% of it used to test the model

  let model = DecisionTree::params().fit(&train).unwrap(); //creates model
  
  let predictions = model.predict(&test); //creates predictions
  
  //outputs predictions and test targets to csv files
  data_to_csv("./outputs/predictions.csv", &predictions);
  data_to_csv("./outputs/test_targets.csv", &test.targets);
  // let model = DecisionTree::params().split_quality(SplitQuality::Gini).fit(&dataset).unwrap();
  // helper("./src/test.csv", &dataset);

  //finds the accuracy of the decision tree model
  let accuracy = model.predict(&dataset.clone()).confusion_matrix(&dataset).unwrap().accuracy();
  println!("Accuracy is: {}%", accuracy * 100.0);

  //attempts to create a latex file that contains the steps taken by the decision tree; we have so much data that the latex file cannot be created properly for models with a higher split ratio
  // File::create("./outputs/tree.tex").unwrap().write_all(model.export_to_tikz().with_legend().to_string().as_bytes()).unwrap();
}

fn get_dataset() -> Dataset<f32, usize, ndarray::Dim<[usize; 1]>> {
  let mut reader = Reader::from_path("./src/House-Price-Prediction-clean.csv").unwrap();
  
  let headers = get_headers(&mut reader);
  let data = get_data(&mut reader);
  let target_index = headers.len() - 1;
  
  let features = headers[0..target_index].to_vec();
  let records = get_records(&data, target_index);
  let targets = get_targets(&data, target_index);
  
  return Dataset::new(records, targets)
    .with_feature_names(features);
}

fn get_headers(reader: &mut Reader<File>) -> Vec<String> {
  return reader
    .headers().unwrap().iter()
    .map(|r| r.to_owned())
    .collect();
  }
  
  fn get_records(data: &Vec<Vec<f32>>, target_index: usize) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
      records.extend_from_slice( &record[0..target_index] );
    }
    return Array::from( records ).into_shape((1460, 31)).unwrap();
  }
  
  fn get_targets(data: &Vec<Vec<f32>>, target_index: usize) -> Array1<usize> {
    let targets = data
      .iter()
      .map(|record| record[target_index] as usize)
      .collect::<Vec<usize>>();
    return Array::from( targets );
  }
  
  fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
    return reader
      .records()
      .map(|r|
        r
          .unwrap().iter()
          .map(|field| field.parse::<f32>().unwrap())
          .collect::<Vec<f32>>()
      )
      .collect::<Vec<Vec<f32>>>();
  }

  fn data_to_csv(filename: &str, data: &Array1<usize>) {
    let file = File::create(filename).expect("Unable to create file");
    let mut writer = Writer::from_writer(file);

    for &value in data.iter() {
        writer.write_record(&[value.to_string()]).expect("Unable to write record");
    }

    println!("{} created successfully.", filename);
}

// fn helper(filename: &str, dataset: &Dataset<f32, usize, ndarray::Dim<[usize; 1]>>) {
//   let mut writer = Writer::from_path(filename).expect("Unable to create file");

//   // Write header
//   for feature_name in dataset.feature_names() {
//       writer.write_field(feature_name).expect("Unable to write to CSV");
//   }
//   writer.write_field("target").expect("Unable to write to CSV");
//   writer.write_record(None::<&[u8]>).expect("Unable to write to CSV");

//   // Write data
//   for (features, &target) in dataset.records().outer_iter().zip(dataset.targets().iter()) {
//       for feature_value in features.iter() {
//           writer.write_field(feature_value.to_string()).expect("Unable to write to CSV");
//       }
//       writer.write_field(target.to_string()).expect("Unable to write to CSV");
//       writer.write_record(None::<&[u8]>).expect("Unable to write to CSV");
//   }

//   println!("Dataset written to {} successfully.", filename);
// }