import hydra
import sys
import cv2
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from hydra.utils import instantiate
from loguru import logger
from data_util import load_image_stats
from image_util import draw_coordinate_frame
from transform.affine import Affine

@hydra.main(config_path="config", config_name="train_tn")
def main(cfg: DictConfig) -> None:
    if cfg.debug:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental_run_functions_eagerly(True)
    
    logger.remove()
    logger.add(sys.stderr, level=cfg.log_level)

    data_generator = instantiate(cfg.data_generator)
    model = instantiate(cfg.model)

    image_stats = load_image_stats(f'{cfg.dataset_directory}/scene_0000.hdf5')
    
    model.compile(optimizer=[tf.keras.optimizers.Adam(learning_rate=0.00001),
                             tf.keras.optimizers.Adam(learning_rate=0.00001)],
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    
    for i in range(5):
        model.fit(data_generator, epochs=10)

        inputs = data_generator[0][0][0]
        attention, yaw = model.infer(inputs)
        for i in range(data_generator.batch_size):
            attention_numpy = attention[i].numpy()
            
            # Find position from attention map maximum
            position_max_idx = np.unravel_index(np.argmax(attention_numpy), attention_numpy.shape)
            # Convert pixel coordinates to workspace coordinates
            workspace_width = cfg.workspace_bounds[0][1] - cfg.workspace_bounds[0][0]
            workspace_length = cfg.workspace_bounds[1][1] - cfg.workspace_bounds[1][0]
            x = cfg.workspace_bounds[0][0] + position_max_idx[1] * workspace_width / inputs.shape[1]
            y = cfg.workspace_bounds[1][0] + position_max_idx[0] * workspace_length / inputs.shape[2]
            z = 0.04
            
            orientation_max_idx = np.argmax(yaw[i].numpy())
            yaw_rad = np.pi / model.n_orentation_bins * orientation_max_idx
            rotation = [np.pi, 0, float(yaw_rad)]  # Convert to Euler angles (xyz)
            pose = Affine(translation=[x, y, z], rotation=rotation)
            
            # Visualization code (existing)
            attention_numpy = (attention_numpy - np.min(attention_numpy)) / (np.max(attention_numpy) - np.min(attention_numpy))
            attention_numpy = (attention_numpy * 255).astype(np.uint8)
            attention_numpy = cv2.applyColorMap(attention_numpy, cv2.COLORMAP_JET)
            
            # Draw coordinate frame on colormap using the computed pose
            colormap = inputs[i][..., 1:]
            colormap = colormap * image_stats['std_colormap'] + image_stats['mean_colormap']
            colormap = colormap.astype(np.uint8)
            colormap_with_pose = draw_coordinate_frame(
                colormap.copy(), 
                pose.matrix,
                cfg.workspace_bounds,
                inputs.shape[1:3]
            )
            
            cv2.imshow("attention", attention_numpy)
            cv2.imshow("colormap with pose", colormap_with_pose)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        if key == ord('q'):
            break
    

if __name__ == "__main__":
    main()