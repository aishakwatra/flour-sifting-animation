#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_m.h"
#include "shader_c.h"
#include "camera.h"

#include <iostream>
#include <vector>
#include <random>



void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);


// --- Settings ---
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

// --- Particle System Constants ---
const unsigned int NUM_PARTICLES_X = 15;
const unsigned int NUM_PARTICLES_Y = 15;
const unsigned int NUM_PARTICLES_Z = 15;
const unsigned int TOTAL_PARTICLES = NUM_PARTICLES_X * NUM_PARTICLES_Y * NUM_PARTICLES_Z;

// --- Global Variables ---
unsigned int particlePosSSBO = 0;
unsigned int particleVelSSBO = 0;
unsigned int particleVAO = 0;

// Camera
Camera camera(glm::vec3(0.0f, 5.0f, -40.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// --- NEW: Fixed Time Step ---
const float PHYSICS_TIME_STEP = 0.005f; // Must match DELTA_T in particles.cs
float physicsAccumulator = 0.0f;
// ---

// Movable Spawn Point
glm::vec3 g_SpawnCenter(0.0f, 15.0f, 0.0f); // Initial spawn position
const float SPAWN_RANGE_XZ = 2.0f;
const float CEILING_Y_RANGE = 4.0f;
const float FLOOR_Y = 0.0f;

// --- Lifetime Constants (must match shader) ---
const float MIN_LIFETIME = 3.0;
const float MAX_LIFETIME = 5.0;

void setupParticleBuffers()
{
	std::default_random_engine generator;

	// Spawning constants (MUST MATCH SHADER UNIFORMS/CONSTS)
	std::uniform_real_distribution<float> rand_xz(-SPAWN_RANGE_XZ, SPAWN_RANGE_XZ);
	std::uniform_real_distribution<float> rand_y(FLOOR_Y, g_SpawnCenter.y + CEILING_Y_RANGE);
	std::uniform_real_distribution<float> rand_vel(-8.0f, 0.0f);
	std::uniform_real_distribution<float> rand_life(MIN_LIFETIME, MAX_LIFETIME);


	std::vector<glm::vec4> positions(TOTAL_PARTICLES);
	std::vector<glm::vec4> velocities(TOTAL_PARTICLES);

	for (int i = 0; i < TOTAL_PARTICLES; ++i)
	{
		// Initial spawn respects the g_SpawnCenter
		positions[i] = glm::vec4(
			g_SpawnCenter.x + rand_xz(generator),
			rand_y(generator),
			g_SpawnCenter.z + rand_xz(generator),
			1.0f);
		// Use random lifetime in .w component
		velocities[i] = glm::vec4(0.0f, rand_vel(generator), 0.0f, rand_life(generator));
	}

	size_t bufferSize = positions.size() * sizeof(glm::vec4);

	// Create SSBOs
	glGenBuffers(1, &particlePosSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, particlePosSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, positions.data(), GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particlePosSSBO);

	glGenBuffers(1, &particleVelSSBO);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, particleVelSSBO);
	glBufferData(GL_SHADER_STORAGE_BUFFER, bufferSize, velocities.data(), GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, particleVelSSBO);

	// Setup VAO for Rendering
	glGenVertexArrays(1, &particleVAO);
	glBindVertexArray(particleVAO);
	glBindBuffer(GL_ARRAY_BUFFER, particlePosSSBO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
	glBindVertexArray(0);
}



int main(int argc, char* argv[])
{
	// (Setup omitted for brevity... GLFW, GLAD, etc.)
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Animation - Particle System", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	//input callbacks
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	// Tell GLFW to capture our mouse
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glfwSwapInterval(0); // VSync Off

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background

	// (Query limitations omitted...)
	std::cout << "OpenGL Limitations: " << "..." << std::endl;

	// build and compile shaders
	// -------------------------

	Shader particleRenderShader("particle_render.vs", "particle_render.fs");
	ComputeShader computeShader("particles.cs");

	setupParticleBuffers();


	// render loop
	// -----------
	int fCounter = 0;
	while (!glfwWindowShouldClose(window))
	{
		// Set frame time
		float currentFrame = (float)glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;
		if (fCounter > 500) {
			std::cout << "FPS: " << 1 / deltaTime << std::endl;
			fCounter = 0;
		}
		else {
			fCounter++;
		}

		processInput(window);


		// #########################################
		// ### THE FIX IS HERE ###
		// #########################################

		// 1. Add real-world time to the accumulator
		physicsAccumulator += deltaTime;

		// 2. Run the physics simulation in fixed steps
		// This loop will run 0 times if the frame is too fast,
		// or multiple times if the frame is very slow.
		while (physicsAccumulator >= PHYSICS_TIME_STEP)
		{
			// --- Run ONE step of the simulation ---
			computeShader.use();
			computeShader.setVec3("spawnCenter", g_SpawnCenter);
			computeShader.setFloat("spawnRangeXZ", SPAWN_RANGE_XZ);
			computeShader.setFloat("spawnRangeY", CEILING_Y_RANGE);

			glDispatchCompute(TOTAL_PARTICLES, 1, 1);

			// We need the barrier *inside* the loop to ensure
			// each physics step finishes before the next one starts.
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

			// Subtract the time step we just simulated
			physicsAccumulator -= PHYSICS_TIME_STEP;
		}


		// 3. Render Particles (always happens once per frame)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Setup Camera/Projection/View matrices
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 200.0f);
		glm::mat4 view = camera.GetViewMatrix();

		particleRenderShader.use();
		particleRenderShader.setMat4("projection", projection);
		particleRenderShader.setMat4("view", view);

		// Enable settings for particle visual effect
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_PROGRAM_POINT_SIZE);

		glPointSize(2.0f);
		glBindVertexArray(particleVAO);
		glDrawArrays(GL_POINTS, 0, TOTAL_PARTICLES);
		glBindVertexArray(0);

		glDisable(GL_BLEND);
		glDisable(GL_PROGRAM_POINT_SIZE);

		// --- PARTICLE SYSTEM LOGIC END ---

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// optional: de-allocate all resources once they've outlived their purpose:
	// ------------------------------------------------------------------------

	glDeleteBuffers(1, &particlePosSSBO);
	glDeleteBuffers(1, &particleVelSSBO);
	glDeleteVertexArrays(1, &particleVAO);
	glDeleteProgram(particleRenderShader.ID);
	glDeleteProgram(computeShader.ID);


	glfwTerminate();

	return EXIT_SUCCESS;
}

// (processInput, mouse_callback, etc. are all unchanged)
// ...
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	// Pass deltaTime to camera processor
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);

	// --- Spawn Point Movement (Arrow Keys) ---
	float spawnMoveSpeed = 10.0f * deltaTime; // Speed of the spawn point
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		g_SpawnCenter.z -= spawnMoveSpeed; 1; // Move "forward" in the world
	if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
		g_SpawnCenter.z += spawnMoveSpeed; // Move "backward"
	if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
		g_SpawnCenter.x -= spawnMoveSpeed;
	if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
		g_SpawnCenter.x += spawnMoveSpeed;
}

// glfw: whenever the mouse moves
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
	float xpos = static_cast<float>(xposIn);
	float ypos = static_cast<float>(yposIn);

	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}

	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(static_cast<float>(yoffset));
}