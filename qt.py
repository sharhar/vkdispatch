import sys
from PyQt5.QtWidgets import QApplication, QOpenGLWidget
from PyQt5.QtGui import QOpenGLContext

class OpenGLWidget(QOpenGLWidget):
    def initializeGL(self):
        # Initialize OpenGL here

        # Get the OpenGL functions object
        self.gl = QOpenGLContext.currentContext()

        for ext in self.gl.extensions():
            name_str = bytes(ext).decode('utf-8')

            if 'memory_object' in name_str:
                print(name_str)
            
            if 'semaphore' in name_str:
                print(name_str)

    def paintGL(self):
        pass
        # Render OpenGL graphics here

if __name__ == '__main__':
    app = QApplication(sys.argv)

    widget = OpenGLWidget()
    widget.show()

    sys.exit(app.exec_())