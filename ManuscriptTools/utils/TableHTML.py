from os.path import exists, join, isfile, splitext, relpath, dirname
from os import listdir
from .DirTools import DirTools
import numpy as np


class TableHTML:
    def __init__(self, title, css_file="style.css", js_file=None):
        self.title = title
        self.css_file = css_file
        self.js_file = js_file
        self.content_dict = dict()
        self.verbose = True
        self.columns = list()
        self.table_html = ""
        self.html = ""
        self.file_dir_path = ""

    def add_head(self, column_names):
        if len(self.content_dict) == 0:
            for column_name in column_names:
                self.content_dict[column_name] = list()
            self.columns = column_names
        elif self.verbose:
            print("Error: columns already added")

    def add_row(self, images_info):
        if len(images_info) != len(self.columns):
            print("Error: the entry doesn't match the number of columns")
        for i, image_info in enumerate(images_info):
            self.content_dict[self.columns[i]].append(image_info)

    def get_relative_path(self, src):
        return relpath(src, self.file_dir_path).replace("\\", "/")

    def fig_container(self, idx, src, layout=0):
        src = self.get_relative_path(src)
        image = '<img src="{}" alt="" />'.format(src)
        image_with_link = '<a href="{}" target="_blank">{}</a>'.format(src, image)
        if layout == 0:  # 2 rows
            html = '<table>\n'
            if idx != "":
                html += '   <tr>\n'
                html += '       <td>{}</td>\n'.format(idx)
                html += '   </tr>\n'
            html += '   <tr>\n'
            html += '       <td class="figure">{}</td>\n'.format(image_with_link)
            html += '   </tr>\n'
            html += '</table>'
        else:  # 2 columns
            html = '<table>\n'
            html += '   <tr>\n'
            if idx != "":
                html += '       <td>{}</td>\n'.format(idx)
            html += '       <td class="figure">{}</td>\n'.format(image_with_link)
            html += '   </tr>\n'
            html += '</table>\n'

        return html

    def create_figure(self, image_info):
        return self.fig_container(str(image_info[0]), image_info[1])

    def build_table(self):
        html = ""
        html += "<table class=\"table\" cellspacing=\"0\">\n"
        html += "<tr class=\"row\">\n"
        for column_name in self.content_dict:
            html += "<th class=\"column\">{}</th>\n".format(column_name)

        html += "</tr>\n"
        html += "<tbody>\n"
        content_list = list(self.content_dict.values())
        for i in range(len(content_list[0])):
            html += "<tr class=\"row\">\n"
            for j in range(len(content_list)):
                row_content = content_list[j][i]
                html += "<td class=\"column\">\n"
                if type(row_content) == list:
                    for element in row_content:
                        html += self.create_figure(element)
                else:
                    html += self.create_figure(row_content)
                html += "</td>\n"
            html += "</tr>\n"
        html += "</tbody>\n"
        html += "</table>\n"
        self.table_html = html

    def create_web_page(self, table_html, title, css_file, js_file=None):
        self.create_css_file(css_file)
        html = "<html>\n"
        html += "<head>\n"
        html += "<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">\n".format(
            css_file)
        if not(js_file is None):
            html += "<script type=\"text/javascript\" src=\"{}\"></script>".format(
                js_file)
        html += "<meta charset=\"UTF-8\">\n"
        html += "</head>\n"
        html += "<body>\n"
        html += "<span class=\"page_title\">{}</span>\n".format(title)
        html += "{}\n".format(table_html)
        html += "</body>\n"
        html += "</html>"
        return html

    def save(self, file_path):
        self.file_dir_path = dirname(file_path)
        self.build_table()
        f = open(file_path, "w")
        f.write(self.create_web_page(self.table_html, self.title, self.css_file, self.js_file))
        f.close()

    def create_css_file(self, file_path):
        if not(exists(join(self.file_dir_path, file_path))):
            content = """body{
                            margin: auto;
                            font-size: 20pt;
                            text-align: center;
                        }
                        table{
                            display: inline-table;
                            white-space: nowrap;
                            vertical-align: middle;
                            padding: 5px;
                        }
                        tr{
                            text-align: center;
                        }
                        td{
                            text-align: center;
                            border: 1px solid;
                        }
                        img{
                            max-width:600px;
                            width:100%;
                            height:auto;
                        }
                        .column {
                          border: 1px solid;
                        }
                        .page_title{
                            display:block;
                            margin: 2%;
                            font-weight: bold;
                            font-size: 28pt;
                        }
                        .figure{
                            height: auto;
                            width: 125px;
                            display: inline-block;
                        }
                        .section{
                            font-size: 28pt;
                        }"""
            f = open(join(self.file_dir_path, file_path), "w")
            f.write(content)
            f.close()


if __name__ == "__main__":

    def get_web_path(path, start):
        return relpath(path, start).replace("\\", "/")

    def create_table():
        m1 = DirTools.list_folder_images("D:/Stage/tmp_manuscripts/D3__/illustration")
        m2 = DirTools.list_folder_images("D:/Stage/tmp_manuscripts/D4__/illustration")
        m3 = DirTools.list_folder_images("D:/Stage/tmp_manuscripts/D5__/illustration")
        m4 = DirTools.list_folder_images("D:/Stage/tmp_manuscripts/D2/illustration")
        table = TableHTML('This is a HTML Table!')
        table.add_head(["D3", "D4", "D5", "D2_"])

        def generate_images_info(images_path_list, n):
            return [(idx, images_path_list[idx])
                    for idx in np.random.randint(len(images_path_list), size=n)]

        for i in range(30):
            table.add_row(
                [generate_images_info(m1, 2),
                 generate_images_info(m2, 3),
                 generate_images_info(m3, 2),
                 generate_images_info(m4, 4)])

        table.save("../web_content/lol.html")

    create_table()
    exit()