import torch
import os
import configs
from helper.log import Logging
from helper.mesh.mesh_converter import convert_mesh_to_point_cloud_folder, save_point_cloud_in_pcd
import numpy as np
import random
from handler.predict import Predict
from handler.train_model import TrainModel

random.seed(configs.SEED)
np.random.seed(configs.SEED)
torch.manual_seed(configs.SEED)
torch.cuda.manual_seed(configs.SEED)
torch.cuda.manual_seed_all(configs.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Main:
    def __init__(self, config:configs.BaseConfig):
        self.config = config
        self.logger = Logging(config=config)

    def make_dataset(self, input_dir, out_dir, total_num_points) -> None:
        os.makedirs(out_dir, exist_ok=True)

        processed_count = 0
        
        for dir_path, subfolders, filenames in os.walk(input_dir):
            if not filenames:
                continue

            clean_name = os.path.basename(dir_path).replace(" ", "_").replace("(", "").replace(")", "")

            filename = clean_name + ".pcd"

            try:
                points, colors_int, obj_path = convert_mesh_to_point_cloud_folder(dir_path, total_num_points=total_num_points, visualize=True)
                save_point_cloud_in_pcd(points, colors_int, out_dir, filename=filename)
                processed_count += 1

            except Exception as e:
                print(f"- Failed to process {dir_path}: {e}")

        print(f"\n{'='*60}")
        print(f"- ALL DONE! Processed {processed_count} meshes")
        print(f"{'='*60}")

    def main(self):
        while True:
            self.logger.print("\n=== Menu ===")
            self.logger.print("1. Train Model")
            self.logger.print("2. Test Model")
            self.logger.print("3. Cleaning Object")
            self.logger.print("4. Scaling Object")
            self.logger.print("5. Mesh to point cloud")

            choice = input("Nr: ").strip()

            if not choice:
                break
            elif choice == "1":
                resume = input("Resume training? (y/n): ").strip().lower() == 'y'
                TrainModel(self.logger, self.config).train(resume=resume)
            elif choice == "2":
                TrainModel(self.logger, self.config).test()
            elif choice == "3":
                input_dir_path = input("Input Dir Path: ").strip('"').strip()
                output_dir_path = input("Output Dir Path: ").strip('"').strip()
                plot_dir_path = input("Plot Dir Path: ").strip('"').strip() or None
                keep_label = int(input("Keep Label ID: "))
                headless = input("Headless (y/n): ").lower().strip() == "y"
                Predict(self.config).cleaning_object(input_dir_path, output_dir_path, keep_label, plot_dir=plot_dir_path, headless=headless)
            elif choice == "4":
                input_dir_path = input("Input Dir Path: ").strip('"').strip()
                output_dir_path = input("Output Dir Path: ").strip('"').strip()
                plot_dir_path = input("Plot Dir Path: ").strip('"').strip() or None
                real_marker_pair_length = float(input("Marker Pair Length (cm): "))
                real_marker_pair_center_distance = float(input("Marker Pair Center Distance (cm): "))
                headless = input("Headless (y/n): ").lower().strip() == "y"
                Predict(self.config).scaling_object(input_dir_path, output_dir_path, real_marker_pair_length, real_marker_pair_center_distance, plot_dir=plot_dir_path, headless=headless)
            elif choice == "5":
                print("""Expected folder structure:
                input_dir/
                    data_1/
                        - mesh.obj
                        - mesh.mtl
                        - mesh_tex0.png
                    data_2/
                        - mesh.obj
                        - mesh.mtl
                        - mesh_tex0.png
                    ...
                """)
                input_dir = input('Input directory: ')
                out_dir = input('Output directory: ')
                total_num_points = int(input(f'Total Points ({self.config.total_num_points}): '))
                self.make_dataset(input_dir, out_dir, total_num_points)

if __name__ == "__main__":
    for i, conf in enumerate(configs.CONFIG_LIST):
        print(f"{i}: {conf.name}")

    selected_config = int(input("ID: "))
    Main(configs.CONFIG_LIST[selected_config]).main()