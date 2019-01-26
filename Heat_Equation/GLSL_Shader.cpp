#include <fstream>


#ifdef _WIN32
#include <windows.h>
#endif

#include <GL/glew.h>

#include "GLSL_Shader.h"



GLSLProgram::GLSLProgram(const std::string& vertexShader, const std::string& fragmentShader)
{
    m_vertexShader.filename = vertexShader;
    m_fragmentShader.filename = fragmentShader;
}

GLSLProgram::~GLSLProgram()
{

}

void GLSLProgram::Unload()
{
    glDetachShader(m_programID, m_vertexShader.id);
    glDetachShader(m_programID, m_fragmentShader.id);
    glDeleteShader(m_vertexShader.id);
    glDeleteShader(m_fragmentShader.id);
    glDeleteShader(m_programID);
}

bool GLSLProgram::Initialize()
{
    m_programID = glCreateProgram();
    m_vertexShader.id = glCreateShader(GL_VERTEX_SHADER);
    m_fragmentShader.id = glCreateShader(GL_FRAGMENT_SHADER);

    m_vertexShader.source = ReadFile(m_vertexShader.filename);
    m_fragmentShader.source = ReadFile(m_fragmentShader.filename);

    if (m_vertexShader.source.empty() || m_fragmentShader.source.empty())
    {
        return false;
    }

    const GLchar* tmp = static_cast<const GLchar*>(m_vertexShader.source.c_str());
    glShaderSource(m_vertexShader.id, 1, (const GLchar**)&tmp, NULL);

    tmp = static_cast<const GLchar*>(m_fragmentShader.source.c_str());
    glShaderSource(m_fragmentShader.id, 1, (const GLchar**)&tmp, NULL);

    if (!CompileShader(m_vertexShader) || !CompileShader(m_fragmentShader))
    {
		printf("Shader-Fehler: Shaderprogram konnte nicht richtig kompiliert werden!");
        return false;
    }

    glAttachShader(m_programID, m_vertexShader.id);
    glAttachShader(m_programID, m_fragmentShader.id);

    glLinkProgram(m_programID);
    return true;
}

void GLSLProgram::LinkProgram()
{
	glLinkProgram(m_programID);
}

GLuint GLSLProgram::GetUniformLocation(const string& name)
{
    map<string, GLuint>::iterator i = m_uniformMap.find(name);
    if (i == m_uniformMap.end())
    {
        GLuint location = glGetUniformLocation(m_programID, name.c_str());
        m_uniformMap.insert(std::make_pair(name, location));
        return location;
    }

    return (*i).second;
}

GLuint GLSLProgram::GetAttribLocation(const string& name)
{
    map<string, GLuint>::iterator i = m_attribMap.find(name);
    if (i == m_attribMap.end())
    {
        GLuint location = glGetAttribLocation(m_programID, name.c_str());
        m_attribMap.insert(std::make_pair(name, location));
        return location;
    }

    return (*i).second;
}


void GLSLProgram::SendUniform(const string& name, const int id)
{
    GLuint location = GetUniformLocation(name);
    glUniform1i(location, id);
}

void GLSLProgram::SendUniform4x4(const string& name, const float* matrix, bool transpose)
{
    GLuint location = GetUniformLocation(name);
    glUniformMatrix4fv(location, 1, transpose, matrix);
}

void GLSLProgram::SendUniform3x3(const string& name, const float* matrix, bool transpose)
{
    GLuint location = GetUniformLocation(name);
    glUniformMatrix3fv(location, 1, transpose, matrix);
}

void GLSLProgram::SendUniform2x2(const string& name, const float* matrix, bool transpose)
{
    GLuint location = GetUniformLocation(name);
	glUniformMatrix2fv(location, 1, transpose, matrix);
}

void GLSLProgram::SendUniform(const string& name, const float red, const float green,
                 const float blue, const float alpha)
{
    GLuint location = GetUniformLocation(name);
    glUniform4f(location, red, green, blue, alpha);
}

void GLSLProgram::SendUniform(const string& name, const float x, const float y, const float z)
{
    GLuint location = GetUniformLocation(name);
    glUniform3f(location, x, y, z);
}

void GLSLProgram::SendUniform(const string& name, const float x, const float y)
{
    GLuint location = GetUniformLocation(name);
	glUniform2f(location, x, y);
}

void GLSLProgram::SendUniform(const string& name, const float scalar)
{
    GLuint location = GetUniformLocation(name);
    glUniform1f(location, scalar);
}


void GLSLProgram::BindAttrib(unsigned int index, const string& attribName)
{
    glBindAttribLocation(m_programID, index, attribName.c_str());
}

void GLSLProgram::BindShader()
{
    glUseProgram(m_programID);
}

string GLSLProgram::ReadFile(const string& filename)
{
    ifstream fileIn(filename.c_str());

    if (!fileIn.good())
    {
		printf("Shader-Fehler: Shaderdatei konnte nicht geladen werden werden!");
        return string();
    }

    string stringBuffer(std::istreambuf_iterator<char>(fileIn), (std::istreambuf_iterator<char>()));
    return stringBuffer;
}

bool GLSLProgram::CompileShader(const GLSLShader& shader)
{
    glCompileShader(shader.id);
    GLint result = 0xDEADBEEF;
    glGetShaderiv(shader.id, GL_COMPILE_STATUS, &result);

    if (!result)
    {
        OutputShaderLog(shader);
        return false;
    }

    return true;
}

void GLSLProgram::OutputShaderLog(const GLSLShader& shader)
{
    //vector<char> infoLog;
    GLint infoLen;
	glGetShaderiv(shader.id, GL_INFO_LOG_LENGTH, &infoLen);
	//infoLog.resize(infoLen);
	string infoLog=string("Fehler im ShaderProgramm: \n");
	infoLog.append(shader.filename+"\n");

	char* charArray=new char[infoLen];
	glGetShaderInfoLog(shader.id, infoLen, &infoLen, &charArray[0]);
	
	infoLog.append(charArray);

	printf("Folgender Shader-Fehler trat auf: %s", infoLog.c_str());

	delete charArray;
}