from collections import namedtuple
import dropbox
from pathlib import Path
import json
import os
import contextlib
import time
import datetime

FileDesc = namedtuple('FileDesc', 'name is_dir')

drop_box_prefix = 'datathon/'
data_path = ''

#source for most of this: https://github.com/dropbox/dropbox-sdk-python/blob/master/example/updown.py
class DropBox():
    try:
        dbx = dropbox.Dropbox(json.loads(Path(os.path.join(str(Path.home()), 
                                                           '.dropbox.json')).read_text())['token'])
    except:
        raise RuntimeError("Dropbox token must be in json file at ~/.dropbox.json")
        

    @contextlib.contextmanager
    def stopwatch(message):
        """Context manager to print how long a block of code took."""
        t0 = time.time()
        try:
            yield
        finally:
            t1 = time.time()
            print('Total elapsed time for %s: %.3f' % (message, t1 - t0))

    @staticmethod
    def upload(path, overwrite=True):
        full_path = data_path + path
        if os.path.isdir(full_path):
            walk = os.walk(full_path)
        elif os.path.exists(full_path):
            walk = [(os.path.dirname(full_path), None, [os.path.basename(full_path)])]
        else:
            raise RuntimeError("No such file")
        for dn, _, files in walk:
            for f in files:
                DropBox._upload(os.path.join(dn, f), drop_box_prefix, dn, f, overwrite=overwrite)

    @staticmethod
    def _upload(fullname, folder, subfolder, name, overwrite=True):
        """Upload a file.
        Return the request response, or None in case of error.
        """
        path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
        while '//' in path:
            path = path.replace('//', '/')
        mode = (dropbox.files.WriteMode.overwrite
                if overwrite
                else dropbox.files.WriteMode.add)
        mtime = os.path.getmtime(fullname)
        with open(fullname, 'rb') as f:
            data = f.read()
        with DropBox.stopwatch('upload %d bytes' % len(data)):
            try:
                res = DropBox.dbx.files_upload(
                    data, path, mode,
                    client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                    mute=True)
            except dropbox.exceptions.ApiError as err:
                print('*** API error', err)
                return None
        print('uploaded as', res.name.encode('utf8'))
        return res

    @staticmethod
    def list(path):
        return list(map(lambda x : FileDesc(name=x[0], is_dir=isinstance(x[1], dropbox.files.FolderMetadata)), 
                        DropBox._list_folder(drop_box_prefix, data_path + path).items()))

    @staticmethod
    def _list_folder(folder, subfolder):
        """List a folder.
        Return a dict mapping unicode filenames to
        FileMetadata|FolderMetadata entries.
        """
        path = '/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'))
        while '//' in path:
            path = path.replace('//', '/')
        path = path.rstrip('/')
        try:
            with DropBox.stopwatch('list_folder'):
                all_res = []
                res = DropBox.dbx.files_list_folder(path)
                all_res.append(res)
                while len(res.entries) != 0:
                    res = DropBox.dbx.files_list_folder_continue(res.cursor)
                    all_res.append(res)
        except dropbox.exceptions.ApiError as err:
            print('Folder listing failed for', path, '-- assumed empty:', err)
            return {}
        else:
            rv = {}
            for r in all_res:
                for entry in r.entries:
                    rv[entry.name] = entry
            return rv

    @staticmethod
    def download(path):
        files = DropBox.list(path)

        if len(files) == 0:
            files = [FileDesc(name=os.path.basename(path), is_dir=False)]
            path = os.path.dirname(path)

        for f in files:
            if f.is_dir:
                DropBox.download(os.path.join(path, f.name))
            else:
                full_save_path = os.path.join(data_path, path)
                if full_save_path != '':
                    os.makedirs(full_save_path, exist_ok=True)
                data = DropBox._download(drop_box_prefix, data_path, os.path.join(path, f.name))
                with open(os.path.join(full_save_path, f.name), 'wb') as f:
                    f.write(data)

    @staticmethod
    def _download(folder, subfolder, name):
        """Download a file.
        Return the bytes of the file, or None if it doesn't exist.
        """
        path = '/%s/%s/%s' % (folder, subfolder.replace(os.path.sep, '/'), name)
        while '//' in path:
            path = path.replace('//', '/')
        with DropBox.stopwatch('download'):
            try:
                md, res = DropBox.dbx.files_download(path)
            except dropbox.exceptions.HttpError as err:
                print('*** HTTP error', err)
                return None
        data = res.content
        print(len(data), 'bytes; md:', md)
        return data

if __name__ == "__main__":
    DropBox.download('data')
