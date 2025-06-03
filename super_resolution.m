low_res_image = imread('test.jpg');

iterations = 100;
lambda_val = 0.1;
rho = 1.0;

R = double(low_res_image(:, :, 1));
G = double(low_res_image(:, :, 2));
B = double(low_res_image(:, :, 3));

x_R = imresize(R, 8, 'bicubic');
x_G = imresize(G, 8, 'bicubic');
x_B = imresize(B, 8, 'bicubic');

z_R = x_R;
z_G = x_G;
z_B = x_B;

u_R = zeros(size(x_R));
u_G = zeros(size(x_G));
u_B = zeros(size(x_B));

for i = 1:iterations
    x_R = update_x(imresize(R, 8, 'bicubic'), z_R, u_R, rho);
    x_G = update_x(imresize(G, 8, 'bicubic'), z_G, u_G, rho);
    x_B = update_x(imresize(B, 8, 'bicubic'), z_B, u_B, rho);
    
    z_R = update_z(x_R, u_R, lambda_val, rho);
    z_G = update_z(x_G, u_G, lambda_val, rho);
    z_B = update_z(x_B, u_B, lambda_val, rho);
    
    u_R = update_u(u_R, x_R, z_R);
    u_G = update_u(u_G, x_G, z_G);
    u_B = update_u(u_B, x_B, z_B);
end

high_res_image = cat(3, uint8(x_R), uint8(x_G), uint8(x_B));

figure;

subplot(1, 2, 1);
imshow(low_res_image);
title('Original Low-Resolution Image');

subplot(1, 2, 2);
imshow(high_res_image);
title('Super-Resolved Image');

imwrite(high_res_image, 'high_resolution_image_color.jpg');

function x = update_x(high_res_initial, z, u, rho)
    x = (high_res_initial + rho * (z - u)) / (1 + rho);
end

function z = update_z(x, u, lambda_val, rho)
    z = clip_image(x + u); 
end

function u = update_u(u, x, z)
    u = u + (x - z);
end

function z = clip_image(image)
    z = max(0, min(255, image));
end
