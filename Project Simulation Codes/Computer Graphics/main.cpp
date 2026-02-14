#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/type_ptr.hpp>
#include <gtc/quaternion.hpp>
#include <iostream>
#include <vector>

const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform int useOrthographic;

out vec3 ourColor;

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
)";

const char* fragmentShaderSource = R"(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(ourColor, 1.0);
}
)";

// Forward declarations
GLuint compileShader(GLenum type, const char* src);
GLuint linkProgram(GLuint vert, GLuint frag);

// 1. Quaternion rotation using matrix conversion
glm::quat rotateQuaternion(glm::quat q, float angle, glm::vec3 axis) {
    glm::quat rotation = glm::angleAxis(glm::radians(angle), axis);
    return rotation * q;
}

// 2. DIRECT QUATERNION ROTATION: p′ = qpq⁻¹
glm::vec3 rotatePointByQuaternion(const glm::vec3& point, const glm::quat& q) {
    // Represent point as pure quaternion: p = (point, 0)
    glm::quat p(0.0f, point.x, point.y, point.z);

    // Calculate inverse quaternion
    glm::quat q_inv = glm::inverse(q);

    // Apply rotation: p′ = q * p * q⁻¹
    glm::quat p_rotated = q * p * q_inv;

    // Extract vector part
    return glm::vec3(p_rotated.x, p_rotated.y, p_rotated.z);
}

// 3. MANUAL VIEW MATRIX CONSTRUCTION
glm::mat4 constructViewMatrixManual(const glm::vec3& cameraPos, const glm::vec3& target, const glm::vec3& up) {
    // Step 1: Calculate basis vectors (as in Example 26)
    glm::vec3 w = glm::normalize(cameraPos - target);  // Forward direction
    glm::vec3 u = glm::normalize(glm::cross(up, w));   // Right direction
    glm::vec3 v = glm::cross(w, u);                    // Up direction

    // Step 2: Construct view matrix manually
    glm::mat4 view = glm::mat4(1.0f);

    view[0][0] = u.x; view[1][0] = u.y; view[2][0] = u.z; view[3][0] = -glm::dot(u, cameraPos);
    view[0][1] = v.x; view[1][1] = v.y; view[2][1] = v.z; view[3][1] = -glm::dot(v, cameraPos);
    view[0][2] = w.x; view[1][2] = w.y; view[2][2] = w.z; view[3][2] = -glm::dot(w, cameraPos);

    return view;
}

// 4. ORTHOGRAPHIC PROJECTION MATRIX
glm::mat4 constructOrthographicMatrixManual(float left, float right, float bottom, float top, float near, float far) {
    glm::mat4 ortho = glm::mat4(1.0f);

    // As shown in Example 27
    ortho[0][0] = 2.0f / (right - left);
    ortho[1][1] = 2.0f / (top - bottom);
    ortho[2][2] = -2.0f / (far - near);

    ortho[3][0] = -(right + left) / (right - left);
    ortho[3][1] = -(top + bottom) / (top - bottom);
    ortho[3][2] = -(far + near) / (far - near);

    return ortho;
}

// 5. DEMONSTRATE EXAMPLE 29: Quaternion Rotation of P(1,0,0) by 90° about Z
void demonstrateQuaternionExample() {
    std::cout << "=== Example 29: Quaternion Rotation ===" << std::endl;

    // Step 1: Create rotation quaternion
    float angle = glm::radians(90.0f);
    glm::vec3 axis(0.0f, 0.0f, 1.0f);
    glm::quat q = glm::angleAxis(angle / 2.0f, axis); // Note: half angle for rotation quaternion

    std::cout << "Rotation quaternion q: (" << q.w << ", " << q.x << ", " << q.y << ", " << q.z << ")" << std::endl;

    // Step 2: Represent point as quaternion
    glm::vec3 point(1.0f, 0.0f, 0.0f);
    std::cout << "Original point P: (" << point.x << ", " << point.y << ", " << point.z << ")" << std::endl;

    // Step 3 & 4: Rotate using p′ = qpq⁻¹
    glm::vec3 rotatedPoint = rotatePointByQuaternion(point, q);
    std::cout << "Rotated point P': (" << rotatedPoint.x << ", " << rotatedPoint.y << ", " << rotatedPoint.z << ")" << std::endl;

    std::cout << "Expected: (0, 1, 0)" << std::endl;
    std::cout << "=====================================" << std::endl << std::endl;
}

void setupTransformations(GLuint program, const glm::mat4& model, bool useOrthographic) {
    GLint modelLoc = glGetUniformLocation(program, "model");
    GLint viewLoc = glGetUniformLocation(program, "view");
    GLint projLoc = glGetUniformLocation(program, "projection");
    GLint orthoLoc = glGetUniformLocation(program, "useOrthographic");

    // Use manual view matrix construction
    glm::mat4 view = constructViewMatrixManual(
        glm::vec3(0.0f, 0.0f, 5.0f),  // Camera position
        glm::vec3(0.0f, 0.0f, 0.0f),  // Look at point  
        glm::vec3(0.0f, 1.0f, 0.0f)   // Up vector
    );

    glm::mat4 projection;
    if (useOrthographic) {
        // Use manual orthographic projection (Example 27)
        projection = constructOrthographicMatrixManual(-2.0f, 2.0f, -2.0f, 2.0f, 0.1f, 100.0f);
    }
    else {
        // Use perspective projection (Example 28)
        projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
    }

    glUseProgram(program);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1i(orthoLoc, useOrthographic ? 1 : 0);
}

int main() {

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGl Transformation Operations", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGL((GLADloadfunc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_DEPTH_TEST);

    // DEMONSTRATE THE QUATERNION EXAMPLE
    demonstrateQuaternionExample();

    // Build shaders
    GLuint vs = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    if (!vs) return -1;
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    if (!fs) return -1;
    GLuint program = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!program) return -1;

    // Simple cube (positions + colors)
    float vertices[] = {
        // positions         // colors
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.2f, 0.8f, 0.2f,
        -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,

        -0.5f,  0.5f,  0.5f,  0.6f, 0.4f, 0.8f,
        -0.5f,  0.5f, -0.5f,  0.2f, 0.6f, 0.6f,
        -0.5f, -0.5f, -0.5f,  0.8f, 0.3f, 0.3f,
        -0.5f, -0.5f, -0.5f,  0.8f, 0.3f, 0.3f,
        -0.5f, -0.5f,  0.5f,  0.3f, 0.7f, 0.9f,
        -0.5f,  0.5f,  0.5f,  0.6f, 0.4f, 0.8f,

         0.5f,  0.5f,  0.5f,  0.9f, 0.6f, 0.4f,
         0.5f,  0.5f, -0.5f,  0.4f, 0.2f, 0.8f,
         0.5f, -0.5f, -0.5f,  0.2f, 0.5f, 0.7f,
         0.5f, -0.5f, -0.5f,  0.2f, 0.5f, 0.7f,
         0.5f, -0.5f,  0.5f,  0.7f, 0.2f, 0.6f,
         0.5f,  0.5f,  0.5f,  0.9f, 0.6f, 0.4f,

        -0.5f, -0.5f, -0.5f,  0.3f, 0.3f, 0.7f,
         0.5f, -0.5f, -0.5f,  0.7f, 0.3f, 0.3f,
         0.5f, -0.5f,  0.5f,  0.4f, 0.9f, 0.4f,
         0.5f, -0.5f,  0.5f,  0.4f, 0.9f, 0.4f,
        -0.5f, -0.5f,  0.5f,  0.8f, 0.3f, 0.9f,
        -0.5f, -0.5f, -0.5f,  0.3f, 0.3f, 0.7f,

        -0.5f,  0.5f, -0.5f,  0.5f, 0.8f, 0.2f,
         0.5f,  0.5f, -0.5f,  0.2f, 0.9f, 0.6f,
         0.5f,  0.5f,  0.5f,  0.9f, 0.2f, 0.5f,
         0.5f,  0.5f,  0.5f,  0.9f, 0.2f, 0.5f,
        -0.5f,  0.5f,  0.5f,  0.4f, 0.6f, 0.9f,
        -0.5f,  0.5f, -0.5f,  0.5f, 0.8f, 0.2f
    };

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    constexpr GLsizei stride = 6 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Quaternion-based orientation
    glm::quat orientation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    bool useOrthographic = false;
    double lastToggleTime = glfwGetTime();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Toggle between orthographic and perspective with space bar
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS &&
            glfwGetTime() - lastToggleTime > 0.5) {
            useOrthographic = !useOrthographic;
            std::cout << "Switched to " << (useOrthographic ? "Orthographic" : "Perspective")
                << " projection" << std::endl;
            lastToggleTime = glfwGetTime();
        }

        // Clear
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.12f, 0.12f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Animate rotation using quaternion helper
        static double lastTime = glfwGetTime();
        double now = glfwGetTime();
        float dt = float(now - lastTime);
        lastTime = now;
        float frameAngle = 45.0f * dt;
        orientation = rotateQuaternion(orientation, frameAngle, glm::vec3(0.0f, 1.0f, 0.0f));
        orientation = glm::normalize(orientation);

        // Build model matrix using quaternion rotation
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::scale(model, glm::vec3(1.5f, 1.5f, 1.5f));
        glm::mat4 rotMat = glm::mat4_cast(orientation);
        model = rotMat * model;
        model = glm::rotate(model, glm::radians(45.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        // Upload transformations with projection toggle
        setupTransformations(program, model, useOrthographic);

        glUseProgram(program);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteProgram(program);
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

// Simple shader helpers
GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);

    GLint success = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint len = 0;
        glGetShaderiv(s, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetShaderInfoLog(s, len, nullptr, log.data());
        std::cerr << "Shader compile error: " << log.data() << std::endl;
        glDeleteShader(s);
        return 0;
    }
    return s;
}

GLuint linkProgram(GLuint vert, GLuint frag) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vert);
    glAttachShader(p, frag);
    glLinkProgram(p);

    GLint success = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &success);
    if (!success) {
        GLint len = 0;
        glGetProgramiv(p, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len);
        glGetProgramInfoLog(p, len, nullptr, log.data());
        std::cerr << "Program link error: " << log.data() << std::endl;
        glDeleteProgram(p);
        return 0;
    }
    return p;
}

