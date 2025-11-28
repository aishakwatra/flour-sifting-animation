#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "shader_m.h"
#include "shader_c.h"
#include "camera.h"
#include "model.h"
#include <stb_image.h>
#include "Skybox.h"

#include <iostream>
#include <vector>
#include <random>

#include <cfloat> // for FLT_MAX

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);


// --- Settings ---
const unsigned int SCR_WIDTH = 1800;
const unsigned int SCR_HEIGHT = 1300;

// --- Particle System Constants ---
const unsigned int NUM_PARTICLES_X = 40;
const unsigned int NUM_PARTICLES_Y = 40;
const unsigned int NUM_PARTICLES_Z = 40;
const unsigned int TOTAL_PARTICLES = NUM_PARTICLES_X * NUM_PARTICLES_Y * NUM_PARTICLES_Z;

// --- Global Variables ---
unsigned int particlePosSSBO = 0;
unsigned int particleVelSSBO = 0;
unsigned int particleVAO = 0;

unsigned int flourHeightTex = 0;

// Camera
Camera camera(glm::vec3(0.0f, 5.0f, 50.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f, 0.0f);
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

// Timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

// --- NEW: Fixed Time Step ---
const float PHYSICS_TIME_STEP = 0.005f; 
float physicsAccumulator = 0.0f;
// ---

//Spawn Point
glm::vec3 g_SpawnCenter(0.0f, 14.5f, 0.0f);
const float SPAWN_RANGE_XZ = 5.0f;
const float FLOOR_Y =-4.0f; 


//HEIGHTMAP CONSTRAINTS
const int HM_WIDTH = 700;
const int HM_HEIGHT = 700;

// base plane for flour/table
const float TABLE_Y = -2.3f;

const float TABLE_SIZE_X = 30.0f; // covers [-15, 15] in X
const float TABLE_SIZE_Z = 30.0f; // covers [-15, 15] in Z

const float TABLE_MIN_X = -TABLE_SIZE_X * 0.5f;
const float TABLE_MAX_X = TABLE_SIZE_X * 0.5f;
const float TABLE_MIN_Z = -TABLE_SIZE_Z * 0.5f;
const float TABLE_MAX_Z = TABLE_SIZE_Z * 0.5f;


unsigned int flourVAO = 0;
unsigned int flourVBO = 0;
unsigned int flourEBO = 0;
unsigned int flourIndexCount = 0;

const int FLOUR_GRID_W = HM_WIDTH;
const int FLOUR_GRID_H = HM_HEIGHT;

const float flourUnitHeight = 0.00005f;

//SIFT CONSTANTS
glm::vec3 g_SifterCenter(0.0f, FLOOR_Y + 15.0f, 0.0f);
const float SIFTER_SCALE = 4.0f;

glm::vec3 kitchenCenter(0.0f, -40.0f, -20.0f);

//side-to-side animation parameters
const float SIFTER_AMPLITUDE = 3.0f;   // how far left/right it moves
const float SIFTER_SPEED = 15.0f;   // oscillation speed

const float SIFTER_JITTER_AMPLITUDE = 0.5f; // how much it wiggles sideways

glm::vec2 g_SifterDirXZ(1.0f, 0.5f);   
glm::vec2 g_SifterDirPerpXZ;

struct SifterBounds
{
	glm::vec3 centerModel;  // center of bounding box in model space
	float     radiusXZ;     // radius in XZ (for the circular bound)
	float     minY;         // vertical extents if you ever need them
	float     maxY;
};

SifterBounds g_SifterBoundsModel;


void setupFlourHeightmap()
{
	glGenTextures(1, &flourHeightTex);
	glBindTexture(GL_TEXTURE_2D, flourHeightTex);

	// Allocate and initialize to 0.0f
	std::vector<float> zeros(HM_WIDTH * HM_HEIGHT, 0.0f);

	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_R32UI,
		HM_WIDTH,
		HM_HEIGHT,
		0,
		GL_RED_INTEGER,
		GL_UNSIGNED_INT,
		zeros.data()
	);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);

}

SifterBounds ComputeSifterBounds(const Model& model)
{
	SifterBounds result{};

	glm::vec3 minB(FLT_MAX);
	glm::vec3 maxB(-FLT_MAX);

	for (const Mesh& mesh : model.meshes)
	{
		for (const auto& v : mesh.vertices)
		{
			const glm::vec3& p = v.Position;
			minB = glm::min(minB, p);
			maxB = glm::max(maxB, p);
		}
	}

	glm::vec3 center = 0.5f * (minB + maxB);

	// 2) Radius in XZ from this center (tightest circle including side walls)
	float radiusXZ = 0.0f;
	for (const Mesh& mesh : model.meshes)
	{
		for (const auto& v : mesh.vertices)
		{
			glm::vec3 p = v.Position;
			glm::vec2 d = glm::vec2(p.x - center.x, p.z - center.z);
			float r = glm::length(d);
			if (r > radiusXZ)
				radiusXZ = r;
		}
	}

	result.centerModel = center;
	result.radiusXZ = radiusXZ;
	result.minY = minB.y;
	result.maxY = maxB.y;

	return result;
}


void setupFlourMesh()
{
	struct FlourVertex {
		glm::vec2 xz;
		glm::vec2 uv;
	};

	std::vector<FlourVertex> vertices;
	vertices.reserve(FLOUR_GRID_W * FLOUR_GRID_H);

	std::vector<unsigned int> indices;
	indices.reserve((FLOUR_GRID_W - 1) * (FLOUR_GRID_H - 1) * 6);

	// Build vertices
	for (int j = 0; j < FLOUR_GRID_H; ++j)
	{
		float v = (FLOUR_GRID_H > 1) ? float(j) / float(FLOUR_GRID_H - 1) : 0.0f;
		float z = TABLE_MIN_Z + v * (TABLE_MAX_Z - TABLE_MIN_Z);

		for (int i = 0; i < FLOUR_GRID_W; ++i)
		{
			float u = (FLOUR_GRID_W > 1) ? float(i) / float(FLOUR_GRID_W - 1) : 0.0f;
			float x = TABLE_MIN_X + u * (TABLE_MAX_X - TABLE_MIN_X);

			FlourVertex vert;
			vert.xz = glm::vec2(x, z);
			vert.uv = glm::vec2(u, v);
			vertices.push_back(vert);
		}
	}

	// Build indices (two triangles per quad)
	for (int j = 0; j < FLOUR_GRID_H - 1; ++j)
	{
		for (int i = 0; i < FLOUR_GRID_W - 1; ++i)
		{
			int row1 = j * FLOUR_GRID_W;
			int row2 = (j + 1) * FLOUR_GRID_W;

			unsigned int i0 = row1 + i;
			unsigned int i1 = row1 + i + 1;
			unsigned int i2 = row2 + i;
			unsigned int i3 = row2 + i + 1;

			// Triangle 1: i0, i2, i1
			indices.push_back(i0);
			indices.push_back(i2);
			indices.push_back(i1);

			// Triangle 2: i1, i2, i3
			indices.push_back(i1);
			indices.push_back(i2);
			indices.push_back(i3);
		}
	}

	flourIndexCount = static_cast<unsigned int>(indices.size());

	// Upload to GPU
	glGenVertexArrays(1, &flourVAO);
	glGenBuffers(1, &flourVBO);
	glGenBuffers(1, &flourEBO);

	glBindVertexArray(flourVAO);

	glBindBuffer(GL_ARRAY_BUFFER, flourVBO);
	glBufferData(GL_ARRAY_BUFFER,
		vertices.size() * sizeof(FlourVertex),
		vertices.data(),
		GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, flourEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER,
		indices.size() * sizeof(unsigned int),
		indices.data(),
		GL_STATIC_DRAW);

	// layout(location=0): vec2 xz
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
		sizeof(FlourVertex),
		(void*)offsetof(FlourVertex, xz));

	// layout(location=1): vec2 uv
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
		sizeof(FlourVertex),
		(void*)offsetof(FlourVertex, uv));

	glBindVertexArray(0);
}



void setupParticleBuffers()
{
	std::default_random_engine generator;
	std::uniform_real_distribution<float> rand_vel(-8.0f, 0.0f);

	std::uniform_real_distribution<float> rand01(0.0f, 1.0f);

	std::vector<glm::vec4> positions(TOTAL_PARTICLES);
	std::vector<glm::vec4> velocities(TOTAL_PARTICLES);

	for (unsigned int i = 0; i < TOTAL_PARTICLES; i++)
	{
		float u = rand01(generator);
		float v = rand01(generator);

		float radius = std::sqrt(u) * SPAWN_RANGE_XZ;
		float angle = 2.0f * 3.14159265f * v;

		float dx = std::cos(angle) * radius;
		float dz = std::sin(angle) * radius;

		positions[i] = glm::vec4(
			g_SpawnCenter.x + dx,
			g_SpawnCenter.y,
			g_SpawnCenter.z + dz,
			1.0f
		);

		velocities[i] = glm::vec4(
			0.0f,
			rand_vel(generator),  
			0.0f,
			0.0f
		);
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
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Flour Sifting Animation", NULL, NULL);
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

	stbi_set_flip_vertically_on_load(false);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background



	// (Query limitations omitted...)
	std::cout << "OpenGL Limitations: " << "..." << std::endl;

	// build and compile shaders
	// -------------------------

	Shader particleRenderShader("particle_render.vs", "particle_render.fs");
	ComputeShader computeShader("particles.cs");
	Shader modelShader("model_loading.vs", "model_loading.fs");
	Shader flourShader("flour.vs", "flour.fs");
	Shader skyboxShader("skybox.vs", "skybox.fs");
	std::vector<std::string> faces = {
		"Textures/skybox/sunset/px.jpg",
		"Textures/skybox/sunset/nx.jpg",
		"Textures/skybox/sunset/py.jpg",
		"Textures/skybox/sunset/ny.jpg",
		"Textures/skybox/sunset/pz.jpg",
		"Textures/skybox/sunset/nz.jpg"
	};

	Skybox skybox(faces, skyboxShader.getID());

	Model tableModel("Objects/table/table.obj");
	Model sieveModel("Objects/sieve/sieve.obj");
	Model KitchenScene("Objects/Kitchen Scene/kitchen.obj");

	g_SifterBoundsModel = ComputeSifterBounds(sieveModel);

	g_SifterDirXZ = glm::normalize(g_SifterDirXZ);
	g_SifterDirPerpXZ = glm::vec2(-g_SifterDirXZ.y, g_SifterDirXZ.x);

	setupParticleBuffers();
	setupFlourHeightmap();
	setupFlourMesh();

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


		// --- Sifter motion (shake in XZ) ---
		float tMain = std::sin(currentFrame * SIFTER_SPEED + 1.0f); // [-1, 1]
		glm::vec2 offsetMain = g_SifterDirXZ * (SIFTER_AMPLITUDE * tMain);

		// small perpendicular wobble to avoid a perfect straight line
		float tJitter = std::sin(currentFrame * SIFTER_SPEED * 1.7f + 1.0f); // different phase/freq
		glm::vec2 offsetJitter = g_SifterDirPerpXZ * (SIFTER_JITTER_AMPLITUDE * tJitter);

		// total horizontal offset from center
		glm::vec2 offsetXZ = offsetMain + offsetJitter;

		// apply to sifter center (Y stays fixed)
		g_SifterCenter.x = offsetXZ.x;
		g_SifterCenter.z = offsetXZ.y;

		//// --- Make emitter follow the *bottom* of the sieve ---
		//// Same bottom we use for sifter collisions
		glm::vec3 sifterCenterWorld = g_SifterCenter + g_SifterBoundsModel.centerModel * SIFTER_SCALE;
		float sifterBottomY = g_SifterCenter.y + g_SifterBoundsModel.minY * SIFTER_SCALE;

		// Put spawn disk just above the grid plane so flour appears to emerge from the mesh
		g_SpawnCenter.x = sifterCenterWorld.x;
		g_SpawnCenter.z = sifterCenterWorld.z;
		g_SpawnCenter.y = sifterBottomY + 3.0f;  


		physicsAccumulator += deltaTime;

		while (physicsAccumulator >= PHYSICS_TIME_STEP)
		{
			
			computeShader.use();
			computeShader.setVec3("spawnCenter", g_SpawnCenter);
			computeShader.setFloat("spawnRangeXZ", SPAWN_RANGE_XZ);

			// uniforms for heightmap
			computeShader.setFloat("tableMinX", TABLE_MIN_X);
			computeShader.setFloat("tableMaxX", TABLE_MAX_X);
			computeShader.setFloat("tableMinZ", TABLE_MIN_Z);
			computeShader.setFloat("tableMaxZ", TABLE_MAX_Z);
			computeShader.setFloat("tableY", TABLE_Y);
			computeShader.setInt("hmWidth", HM_WIDTH);
			computeShader.setInt("hmHeight", HM_HEIGHT);

			computeShader.setFloat("flourUnitHeight", flourUnitHeight);

			glBindImageTexture(
				0,                    // image unit
				flourHeightTex,
				0,
				GL_FALSE,
				0,
				GL_READ_WRITE,
				GL_R32UI              // must match internal format
			);


			glm::vec3 sifterCenterWorld = g_SifterCenter + g_SifterBoundsModel.centerModel * SIFTER_SCALE;

			// radius in XZ from bounds
			float sifterRadiusWorld = g_SifterBoundsModel.radiusXZ * SIFTER_SCALE;

			// bottom of the sieve (minY) in world space
			float sifterBottomY = g_SifterCenter.y + g_SifterBoundsModel.minY * SIFTER_SCALE;

			computeShader.setVec3("sifterCenter", sifterCenterWorld);
			computeShader.setFloat("sifterRadius", sifterRadiusWorld);
			computeShader.setFloat("sifterY", sifterBottomY);

			computeShader.setFloat("sifterCellSize", 0.12f);      // spacing
			computeShader.setFloat("sifterBarThickness", 0.1f); // thickness

			glDispatchCompute(TOTAL_PARTICLES, 1, 1);
			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

			physicsAccumulator -= PHYSICS_TIME_STEP;
		}


		// Render Particles (always happens once per frame)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Setup Camera/Projection/View matrices
		glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 200.0f);
		glm::mat4 view = camera.GetViewMatrix();

		modelShader.use();
		modelShader.setMat4("projection", projection);
		modelShader.setMat4("view", view);


		//DRAW TABLE
		glm::mat4 model = glm::mat4(1.0f);
		model = glm::translate(model, glm::vec3(0.0f, FLOOR_Y - 2.5f, 0.0f));
		model = glm::scale(model, glm::vec3(1.0f));
		modelShader.setMat4("model", model);
		tableModel.Draw(modelShader);

		//DRAW KITCHEN
		glm::mat4 modelKitchen = glm::mat4(1.0f);
		modelKitchen = glm::translate(modelKitchen, kitchenCenter);
		modelKitchen = glm::scale(modelKitchen, glm::vec3(4));
		modelShader.setMat4("model", modelKitchen);
		KitchenScene.Draw(modelShader);


		//DRAW SIEVE
		glm::mat4 modelSieve = glm::mat4(1.0f);
		modelSieve = glm::translate(modelSieve, g_SifterCenter);
		modelSieve = glm::scale(modelSieve, glm::vec3(SIFTER_SCALE));
		modelShader.setMat4("model", modelSieve);
		sieveModel.Draw(modelShader);

		//DRAW SKYBOX
		skybox.draw(view, projection);


		//DRAW FLOUR HEIGHTMAP
		flourShader.use();
		flourShader.setMat4("projection", projection);
		flourShader.setMat4("view", view);

		glm::mat4 flourModelMat = glm::mat4(1.0f);
		flourShader.setMat4("model", flourModelMat);

		flourShader.setFloat("tableY", TABLE_Y);
		flourShader.setFloat("flourUnitHeight", flourUnitHeight);
		flourShader.setInt("hmWidth", HM_WIDTH);
		flourShader.setInt("hmHeight", HM_HEIGHT);

		flourShader.setVec3("lightDir", glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f)));
		flourShader.setVec3("lightColor", glm::vec3(1.0f));
		flourShader.setVec3("flourColor", glm::vec3(0.98f, 0.98f, 0.95f));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, flourHeightTex);

		flourShader.use();
		flourShader.setInt("flourHeightmap", 0);
		flourShader.setFloat("flourUnitHeight", 0.003f);

		glBindVertexArray(flourVAO);
		glDrawElements(GL_TRIANGLES, flourIndexCount, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		particleRenderShader.use();
		particleRenderShader.setMat4("projection", projection);
		particleRenderShader.setMat4("view", view);

		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glEnable(GL_PROGRAM_POINT_SIZE);

		glPointSize(2.0f);
		glBindVertexArray(particleVAO);
		glDrawArrays(GL_POINTS, 0, TOTAL_PARTICLES);
		glBindVertexArray(0);

		glDisable(GL_BLEND);
		glDisable(GL_PROGRAM_POINT_SIZE);


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
	glDeleteProgram(modelShader.ID);


	glfwTerminate();

	return EXIT_SUCCESS;
}

// (processInput, mouse_callback, etc. are all unchanged)
// ...
void processInput(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	float cameraSpeed = 10.0f * deltaTime;

	// Pass deltaTime to camera processor
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, cameraSpeed);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, cameraSpeed);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, cameraSpeed);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, cameraSpeed);

	// --- Spawn Point Movement (Arrow Keys) ---
	float spawnMoveSpeed = 10.0f * deltaTime; // Speed of the spawn point
	if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
		g_SpawnCenter.z -= spawnMoveSpeed; // Move "forward" in the world
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