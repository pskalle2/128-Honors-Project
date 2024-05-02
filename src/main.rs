use csv::{Reader, Writer};
use std::{fs::File};
use ndarray::{ Array, Array1, Array2};
use linfa::Dataset;
use linfa_trees::{DecisionTree, TreeNode};
use linfa::prelude::*;
use plotters::prelude::*;
use std::io::Write;
use std::process::Command;

fn main() {
  let dataset: DatasetBase<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>, ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>>> = get_dataset("./src/House-Price-Prediction-clean.csv");
  let (train, test) = dataset.clone().split_with_ratio(0.90); //splits the dataset with 90% of it used to train the model and 10% of it used to test the model

  let model: DecisionTree<f32, usize> = DecisionTree::params().fit(&train).unwrap(); //creates model
  
  let predictions: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>> = model.predict(&test); //creates predictions
  
  //outputs predictions and test targets to csv files
  data_to_csv("./outputs/predictions.csv", &predictions);
  data_to_csv("./outputs/test_targets.csv", &test.targets);


  //creates a list of the feautures sorted by importance
  let binding: Vec<String> = dataset
        .feature_names();
  let sorted_feature_importances: Vec<(&str, f64)> = binding
        .iter()
        .map(|s| s.as_str())
        .zip(model.feature_importance().iter().copied().map(|x| x as f64)) 
        .collect();

  let mut sorted_feature_importances = sorted_feature_importances;
  sorted_feature_importances.sort_by(|(_, imp1), (_, imp2)| imp2.partial_cmp(imp1).unwrap());

  // Output the sorted feature importances
  println!("Sorted Feature importances:");
  for (feature, importance) in &sorted_feature_importances {
      println!("{}: {}%", feature, importance * 100.0);
  }

  let mut index = sorted_feature_importances.len();
    for (i, &(_, importance)) in sorted_feature_importances.iter().enumerate() {
        if importance == 0.0 {
            index = i;
            break;
        }
    }
  
  //plots all featurss with nonzero importances as a histogram
  plot_feature_importance(&sorted_feature_importances[..index].to_vec());

  //finds the accuracy of the decision tree model
  let matrix = model.predict(&dataset.clone()).confusion_matrix(&dataset).unwrap();
  let accuracy = matrix.accuracy();
  
  println!("Accuracy is: {}%", accuracy * 100.0);

  //creates a .dot file that creates the decision tree
  let dot_filename: &str = "./outputs/tree.dot";
  let file: Result<File, std::io::Error> = File::create(dot_filename);

  export_tree_to_dot(&model, &mut file.unwrap()).expect("Failed to export decision tree to DOT");

  // Convert DOT file to PNG using Graphviz
  let png_filename: &str = "./outputs/tree.png";
  let output: std::process::Output = Command::new("dot")
    .args(&["-Tpng", dot_filename, "-o", png_filename])
    .output()
    .expect("Failed to execute dot command");


  match output.status.success() {
    true => println!("Decision tree visualization saved as {}", png_filename),
    false => {
      eprintln!("Failed to create decision tree visualization");
      std::process::exit(1);      
    }
  }
}

fn get_dataset(path: &str) -> Dataset<f32, usize, ndarray::Dim<[usize; 1]>> {
  let mut reader = Reader::from_path(path).unwrap();
  
  let headers: Vec<String> = get_headers(&mut reader);
  let data: Vec<Vec<f32>> = get_data(&mut reader);
  let target_index: usize = headers.len() - 1;
  
  let features: Vec<String> = headers[1..target_index].to_vec();
  let records: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> = get_records(&data, target_index);
  let targets: ndarray::ArrayBase<ndarray::OwnedRepr<usize>, ndarray::Dim<[usize; 1]>> = get_targets(&data, target_index);
  
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
      records.extend_from_slice( &record[1..target_index] );
    }
    return Array::from( records ).into_shape((1460, 30)).unwrap();
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

fn plot_feature_importance(feature_importances: &Vec<(&str, f64)>) {
  let root = BitMapBackend::new("./outputs/feature_importance.png", (1500, 900)).into_drawing_area();
  root.fill(&WHITE).unwrap();

  let feature_names: Vec<&str> = feature_importances.iter().map(|(name, _)| *name).collect();
  let importances: Vec<f64> = feature_importances.iter().map(|(_, imp)| *imp).collect();

  let num_features = feature_names.len();
  let binding = feature_names.clone();
  let mut chart = ChartBuilder::on(&root)
    .caption("Feature Importance", ("sans-serif", 30).into_font())
    .margin(5)
    .x_label_area_size(40)
    .y_label_area_size(40)
    .build_cartesian_2d(binding.into_segmented(), 0.0..0.2)
    .unwrap();

chart
    .configure_mesh()
    .x_desc("Features")
    .y_desc("Proportion")
    .x_labels(num_features)
    .draw()
    .unwrap();

let data: Vec<(&str, f64)> = feature_names.iter().map(|&name| name).zip(importances).collect();

chart
    .draw_series(
        Histogram::vertical(&chart)
            .style(RED.mix(0.5).filled())
            .data(data.iter().map(|(name, imp)| (name, *imp))),
    )
    .unwrap()
    .label("Feature Importance")
    .legend(|(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], RED.mix(0.5).filled()));


  chart.configure_series_labels().draw().unwrap();
}

fn export_tree_to_dot(tree: &DecisionTree<f32, usize>, file: &mut File) -> std::io::Result<()> {
  writeln!(file, "digraph decision_tree {{")?;
  writeln!(file,  "    layout = sfdp;")?;
  writeln!(file,  "    concentrate = true;")?;
  writeln!(file,  "    ratio = 0.5;")?;

  // Write the decision tree nodes and edges
  write_decision_tree_node_to_dot(&tree.root_node(), file)?;

  writeln!(file, "}}")
}

fn write_decision_tree_node_to_dot(node: &TreeNode<f32, usize>  , file: &mut File) -> std::io::Result<()> {
  if node.is_leaf() {
      writeln!(file, "    {} [label=\"{}\"];", node.split().0, node.prediction().unwrap())?;
  } else {
      writeln!(
          file,
          "    {} [label=\"Feature {} <= {}\"];",
          node.split().0,
          node.feature_name().unwrap(),
          node.split().1
      )?;
      writeln!(
          file,
          "    {} -> {} [label=\"true\"];",
          node.split().0,
          node.children()[0].clone().unwrap().split().0
      )?;
      writeln!(
          file,
          "    {} -> {} [label=\"false\"];",
          node.split().0,
          node.children()[1].clone().unwrap().split().0
      )?;
      write_decision_tree_node_to_dot(&node.children()[0].clone().unwrap(), file)?;
      write_decision_tree_node_to_dot(&node.children()[1].clone().unwrap(), file)?;
  }
  Ok(())
}