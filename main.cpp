#include <QApplication>
#include "uidemo.h"
#include "appinit.h"
#include <QTextCodec>
#include <QFile>
#include <QColor>
#include <QPalette>
#include <QFont>
#include <string>



int main(int argc, char *argv[])
{
   // QTextCodec::setCodecForLocale(QTextCodec::codecForName(“UTF8”))
    QApplication a(argc, argv);
    //QTextCodec::setCodecForTr(QTextCodec::codecForLocale());

    QTextCodec *codec = QTextCodec::codecForName("gb2312");
    QTextCodec::setCodecForLocale(codec);
    //QTextCodec::setCodecForCStrings(codec);
   // QTextCodec::setCodecForTr(codec);
    QApplication::addLibraryPath("./plugins");


    //加载样式表
    QFile file(":/qss/psblack.css");
    if (file.open(QFile::ReadOnly))
    {
        QString qss = QLatin1String(file.readAll());
        QString paletteColor = qss.mid(20, 7);
        qApp->setPalette(QPalette(QColor(paletteColor)));
        qApp->setStyleSheet(qss);
        file.close();
    }

    a.setFont(QFont("Microsoft Yahei", 9));
    AppInit::Instance()->start();

    UIdemo w;
    w.show();

    return a.exec();
}
