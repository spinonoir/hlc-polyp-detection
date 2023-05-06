from ultralytics import YOLO

if __name__ == '__main__':
  batch_sizes = [8, 16]
  learning_rate_pairs = [[1e-4, 1e-2], [1e-3, 1e-3]]

  for batch_size in batch_sizes:
    for learning_rate_pair in learning_rate_pairs:

      model = YOLO('yolov8x.pt')

      name = "yolov8x_kaidong_bs_{batch_size}_lr0_{learning_rate_start}_lrf_{learning_rate_end}"

      results = model.train(
          data='polyp_datasets.yaml',
          imgsz=384,
          epochs=20,
          batch=batch_size,
          lr0=learning_rate_pair[0],
          lrf=learning_rate_pair[1],
          device='cuda:0',
          cache=True,
          name=name.format(batch_size = str(batch_size), learning_rate_start = str(learning_rate_pair[0]), learning_rate_end = str(learning_rate_pair[1]))
      )

      print("----------------")
      print("TRAINING RESULTS")
      print(results)

      metrics = model.val()
      print("----------------")
      print("VALIDATION METRICS")
      print(metrics)

      model.export()
