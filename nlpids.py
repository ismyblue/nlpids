# Name: nlpids.py
# Author: HuangHao
# Time: 2021/3/6 20:38



http_text = """GET http://localhost:8080/tienda1/publico/anadir.jsp?id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir+al+carrito HTTP/1.1
User-Agent: Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko)
Pragma: no-cache
Cache-control: no-cache
Accept: text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,image/png,*/*;q=0.5
Accept-Encoding: x-gzip, x-deflate, gzip, deflate
Accept-Charset: utf-8, utf-8;q=0.5, *;q=0.5
Accept-Language: en
Host: localhost:8080
Cookie: JSESSIONID=B92A8B48B9008CD29F622A994E0F650D
Connection: close"""

from dataset import load_homemade_dataset, load_csic_2010
from keywords import get_homemade_dataset_keywords_dict, get_csic2010_keywords_dict
from preprocessor import PreProcessor


if __name__ == '__main__':
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = load_csic_2010()
    pre = PreProcessor(get_csic2010_keywords_dict())
    for i in range(x_train.shape[0]):
        print(pre.digital_to_words(x_train[i]))
        if i == 10:
            break

    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = load_homemade_dataset()
    pre = PreProcessor(get_homemade_dataset_keywords_dict())
    for i in range(x_train.shape[0]):
        print(pre.digital_to_words(x_train[i]))
        if i == 10:
            break