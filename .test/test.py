import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

# Define the geometry
x_min, x_max = 0, 0.41
y_min, y_max = 0, 2.2
r = 0.05
xc, yc = 0.2, 0.2

# Define the fluid properties
rho = 1
nu = 0.01
Re = 100

# Define the inflow velocity
def inflow(x, y):
    return 4*1.5*y*(x_max-y)/x_max**2

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # Define the neural network architecture
        self.net_u = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.net_v = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
        self.net_p = nn.Sequential(
            nn.Linear(3, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, y, t):
        # Concatenate the input coordinates and time
        input = torch.cat([x.unsqueeze(1), y.unsqueeze(1), t.unsqueeze(1)], dim=1)

        # Compute the velocity and pressure fields
        u = self.net_u(input)
        v = self.net_v(input)
        p = self.net_p(input)

        return u, v, p

    def loss(self, x, y, t, u, v, p):
        # Compute the differential operators
        u_x, u_y, v_x, v_y, p_x, p_y, p_xx, p_yy = self.compute_gradients(x, y, t, u, v, p)

        # Compute the residual equations
        eq1 = u_x + v_y
        eq2 = u*u_x + v*u_y + p_x - 1/rho*(nu*(2*u_x - v_y) + u*(u_x + v_y))
        eq3 = u*v_x + v*v_y + p_y - 1/rho*(nu*(2*v_y - u_x) + v*(u_x + v_y))

        # Compute the boundary and initial conditions
        bc_wall = torch.mean(torch.square(u[x==x_min]) + torch.square(v[x==x_min]))
        bc_cylinder = torch.mean((u**2 + v**2)[((x-xc)**2 + (y-yc)**2)<r**2])
        bc_inflow = torch.mean(torch.square(u[y==y_max] - inflow(x[y==y_max], y[y==y_max])))
        bc_outflow = torch.mean(v[y==y_min])
        ic = torch.mean()

        # Define the coefficients for the loss function
        lambda_eq = 1
        lambda_bc_wall = 1e3
        lambda_bc_cylinder = 1e3
        lambda_bc_inflow = 1e3
        lambda_bc_outflow = 1e3
        lambda_ic = 1e3

        # Compute the loss function
        loss_eq = torch.mean(torch.square(eq1) + torch.square(eq2) + torch.square(eq3))
        loss_bc_wall = torch.mean(torch.square(u[x == x_min]) + torch.square(v[x == x_min]))
        loss_bc_cylinder = torch.mean((u ** 2 + v ** 2)[((x - xc) ** 2 + (y - yc) ** 2) < r ** 2])
        loss_bc_inflow = torch.mean(torch.square(u[y == y_max] - inflow(x[y == y_max], y[y == y_max])))
        loss_bc_outflow = torch.mean(v[y == y_min])
        loss_ic = torch.mean(torch.square(u[t == 0]) + torch.square(v[t == 0]) + torch.square(p[t == 0]))

        loss = lambda_eq * loss_eq + \
               lambda_bc_wall * loss_bc_wall + \
               lambda_bc_cylinder * loss_bc_cylinder + \
               lambda_bc_inflow * loss_bc_inflow + \
               lambda_bc_outflow * loss_bc_outflow + \
               lambda_ic * loss_ic

        return loss

    def compute_gradients(self, x, y, t, u, v, p):
        # Compute the gradients of velocity and pressure
        u_grad = torch.autograd.grad(u, [x, y], create_graph=True, grad_outputs=torch.ones_like(u))[0]
        u_x = u_grad[:, 0]
        u_y = u_grad[:, 1]
        v_grad = torch.autograd.grad(v, [x, y], create_graph=True, grad_outputs=torch.ones_like(v))[0]
        v_x = v_grad[:, 0]
        v_y = v_grad[:, 1]
        p_grad = torch.autograd.grad(p, [x, y], create_graph=True, grad_outputs=torch.ones_like(p))[0]
        p_x = p_grad[:, 0]
        p_y = p_grad[:, 1]

        # Compute the second-order derivatives of pressure
        p_xx = torch.autograd.grad(p_x, x, create_graph=True, grad_outputs=torch.ones_like(p_x))[0]
        p_yy = torch.autograd.grad(p_y, y, create_graph=True, grad_outputs=torch.ones_like(p_y))[0]

        return u_x, u_y, v_x, v_y, p_x, p_y, p_xx, p_yy

# Define the training data
N = 300
x = torch.linspace(x_min, x_max, N)
y = torch.linspace(y_min, y_max, N)
t = torch.linspace(0, 1, N)
X, Y, T = torch.meshgrid(x, y, t)

# Convert the training data to tensors and reshape
X = torch.flatten(X).unsqueeze(1)
Y = torch.flatten(Y).unsqueeze(1)
T = torch.flatten(T).unsqueeze(1)

# Define the PINN model and optimizer
model = PINN()
optimizer = optim.Adam(model.parameters())

# Train the PINN model
for epoch in range(10000):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    u, v, p = model(X, Y, T)

    # Compute the loss
    loss = model.loss(X, Y, T , u, v, p)

# Backward pass
loss.backward()

# Update the weights
optimizer.step()

# Print the loss
if epoch % 100 == 0:
    print('Epoch [{}/{}], Loss: {:.4e}'.format(epoch, 10000, loss.item()))

# Extract the solution
u, v, p = model(X, Y, T)

# Reshape the solution
u = u.reshape(N, N, N).detach().numpy()
v = v.reshape(N, N, N).detach().numpy()
p = p.reshape(N, N, N).detach().numpy()

# Plot the solution
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].contourf(x, y, u[:, :, 0], levels=50, cmap='jet')
axs[0].set_title('u')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[1].contourf(x, y, v[:, :, 0], levels=50, cmap='jet')
axs[1].set_title('v')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[2].contourf(x, y, p[:, :, 0], levels=50, cmap='jet')
axs[2].set_title('p')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
plt.show()

