# Procedure to render .svs files

1. Navigate to the root of this repository
2. Move all the .svs files that you want to view to the folder `dataset/data/check`
3. From the root, execute the command: ```python src/data/renderer/render_svs.py dataset/data/check```
4. The server will be deployed at `http://127.0.0.1:5000` by default
    - TROUBLESHOOT 1: Ensure that the directory `check` exists in the right path
    - TROUBLESHOOT 2: Try changing the port as another application might have been assigned to it during app development
5. Open the link and you should see the list of files
6. Click on the file you want to view - you can open multiple files at the same time in individual tabs