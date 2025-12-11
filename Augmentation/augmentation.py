import cv2, numpy as np, random


class Augmentor:
    def __init__(self):
        self.techniques = [
            [self.flip, 0.4],
            [self.change_brightness, 0.3],
            [self.rotate, 0.3],
            [self.gaussian_noise, 0.1],
            [self.salt_and_pepper_noise, 0.05]
        ]
        self.height, self.width = 512, 384
        self.center = (self.width // 2, self.height // 2)

    def flip(self, img):
        return cv2.flip(img, 1)

    def change_brightness(self, img):
        hsb = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsb = np.array(hsb, dtype=np.float64)

        brightness_ration = random.uniform(0.5, 1.4)
        hsb[:, :, 2] = hsb[:, :, 2] * brightness_ration
        hsb[:, :, 2] = np.clip(hsb[:, :, 2], 0, 255)

        hsb = np.array(hsb, dtype=np.uint8)
        rgb = cv2.cvtColor(hsb, cv2.COLOR_HSV2BGR)
        return rgb

    def rotate(self, img):
        angle = random.uniform(-30, 30)
        if (abs(angle) < 0.1):
            return img

        rotation_matrix = cv2.getRotationMatrix2D(self.center, angle, 1.0)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (self.height, self.width), borderMode=cv2.INTER_LINEAR)
        return rotated_image

    def gaussian_noise(self, img, mean=0, std=15):
        gauss = np.random.normal(mean, std, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    def salt_and_pepper_noise(self, img, amount=0.005, s_vs_p=0.5):
        noisy = img.copy()

        num_pixels = self.height * self.width

        num_salt = int(num_pixels * amount * s_vs_p)
        num_pepper = int(num_pixels * amount * (1 - s_vs_p))

        coords = (np.random.randint(0, self.width, num_salt // 2),
                  np.random.randint(0, self.height, num_salt // 2))
        noisy[coords[0], coords[1], :] = 255

        coords = (np.random.randint(0, self.width, num_pepper // 2),
                  np.random.randint(0, self.height, num_pepper // 2))
        noisy[coords[0], coords[1], :] = 0

        return noisy

    def augment(self, img):
        image_name = ""
        for technique, prob in self.techniques:
            if random.random() < prob and len(image_name) < 3 and (len(image_name) == 0 or image_name[-1] != 'g'):
                img = technique(img)
                image_name += technique.__name__[0]
        augmented_images = [image_name, img]

        if len(image_name == 0):
            choice = random.choice(self.techniques)
            img = choice(img)
            augmented_images = [image_name, img]


        print(f"Applied augmentations: {image_name}")
        return augmented_images