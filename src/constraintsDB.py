# Mỗi ràng_buộc là tuple (type, params, is_hard, weight, threshold)
# Ví dụ: ("is_presence", ["binh_bong_dua", "hoa_sen"], True, 1.0, None)  # Hard, phải có cả hai
# ("at_least", ["binh_bong_dua"], True, 1.0, 10)  # Hard, ít nhất 10 instances
# ("is_on", ["bong_dua", "trai_dua"], False, 0.5, None)  # Soft, weight 0.5 nếu "bong_dua" on "trai_dua" (có thể dùng spatial check)
# ("confidence_min", ["all"], True, 1.0, 0.7)  # Hard, avg confidence >=0.7
CONSTRAINTS_DB = {
    "Lễ hội thác côn": [
        ("is_presence", ["binh bong dua"], True, 1.0, None),
        ("at_least", ["binh bong dua"], True, 1.0, 10),
        ("is_on", ["bong dua", "trai dua"], False, 0.5, None),
        ("at_least_in_frame", ["binh bong dua"], True, 1.0, 2),  # Ít nhất 2 objects cùng frame
        ("confidence_min", ["binh bong dua"], True, 1.0, 0.8)
    ],
    "Sân Khấu Dù Kê": [
        ("is_on", ["Nguoi bieu dien", "Canh thien nhien trong rung"], False, 0.6, None),
        ("is_on", ["Nguoi bieu dien", "Canh cung dien"], False, 0.6, None),
        ("is_on", ["Nguoi bieu dien", "Canh hang dong"], False, 0.6, None),
        ("is_on", ["Nguoi bieu dien", "Canh bien"], False, 0.6, None),
    ],

    "Nghinh Ông": [
        ("at_least_in_frame", ["Trang phuc bieu dien nghe thuat dan gian", "Xe dieu hanh ruoc ong"], True, 1.0, None),
        ("at_least_in_frame", ["Kieu ruoc ong", "Trang phuc linh"], True, 1.0, 5),
        ("at_least_in_frame", ["Trang phuc linh"], False, 0.75, 3),
        ("is_presence", ["Trang phuc linh", "Ao dai truyen thong"], False, 0.5, None),
        ("is_presence", ["Ban tho ong", "Trang phuc le hoi"], False, 0.5, None),
        ("is_presence", ["Ban tho ong", "Ao dai truyen thong"], False, 0.5, None),
    ],

    "Chợ nổi Cái Răng": [
        ("is_presence", ["Cay beo", "Thuyen"], True, 1.0, None),
        ("at_least_in_frame", ["Cay beo", "Thuyen"], True, 1.0, 5),
        ("is_on", ["Cay beo", "Thuyen"], True, 1.0, None)
    ],

    "Đờn ca tài tử": [
        ("is_presence", ["dan kim"], True, 0.25, None),
        ("is_presence", ["dan guitar"], True, 0.25, None),
        ("is_presence", ["dan bau"], True, 0.25, None),
        ("is_presence", ["dan ty ba"], True, 0.25, None),
        ("is_presence", ["dan tam"], True, 0.25, None),
        ("is_presence", ["sao"], True, 0.25, None),
        ("is_presence", ["dan tranh"],True, 0.25, None)
    ],

    "Nhạc Ngũ Âm người Khmer": [
        ("at_least", ["Ken Srolay pin-piet"], False, 0.2, None),
        ("at_least", ["Dan sat Ro-niet-dek"], False, 0.7, None),
        ("at_least", ["Dan cong Coung-touch Coung-thom"],True, 0.9,None),
        ("at_least", ["Dan thuyen Ro-niet-ek"], True, 1.0, 5),
        ("at_least", ["Trong lon Sakho-thom"], False, 0.6, 2),
        ("at_least", ["Trong nho Kha-so-sompho"], False, 0.6, 2),
        ("at_least_in_frame", ["Dan sat Ro-niet-dek", "Trang phuc Khmer"], False, 0.2, None),
        ("at_least_in_frame", ["Dan cong Coung-touch Coung-thom", "Trang phuc Khmer"], False, 0.2, None),
        ("at_least_in_frame", ["Dan thuyen Ro-niet-ek", "Trang phuc Khmer"], False, 0.2, None),
        ("at_least_in_frame", ["Trong lon Sakho-thom", "Trang phuc Khmer"], False, 0.2, None),
        ("at_least_in_frame", ["Trong nho Kha-so-sompho", "Trang phuc Khmer"], False, 0.2, None)
    ],

    "Lễ hội Kỳ Yên Đình Bình Thủy": [
        ("at_least", ["Nghe si Hat Boi"], False, 0.1, 2),
        ("at_least_in_frame", ["Nghe si Hat Boi"], False, 0.15, 2),
        ("at_least", ["Mua Lan"], False, 0.7, 4),
        ("at_least_in_frame", ["Mua Lan"], False, 0.5, 2),
        ("at_least", ["Ban tho Dinh Binh Thuy"], False, 0.3, 2),
        ("confidence_min", ["Ban tho Dinh Binh Thuy"], False, 0.3, 0.8),
        ("at_least_in_frame", ["Ao dai"], False, 0.5, 2),
        ("at_least_in_frame", ["Le phuc Ky Yen"], False, 0.5, 2),
        ("at_least_in_frame", ["Le vat Ky Yen", "Le phuc Ky Yen", "Ao dai"], False, 0.6, None)
    ],

    "Ooc Bom Bóc": [
        ("confidence_min", ["Ok bom boc"], True, 1.0, 0.9),
        ("confidence_min", ["Ghe ngo"], True, 1.0, 0.7),
        ("is_presence_in_frame", ["Den hoa dang", "Den nuoc"], True, 1.0, None),
        ("is_presence_in_frame", ["Den hoa dang dang thung", "Den nuoc"], True, 1.0, None),
        ("at_least_in_frame", ["Com", "Khoai", "Cong tre"], False, 0.6, None),
        ("at_least_in_frame", ["Com", "Khoai", "Cau con ong"], False, 0.8, None),
        ("at_least_in_frame", ["trong Chhay-dam", "Nguoi mua"], False, 0.5, None),
        ("at_least", ["Chay"], False, 0.3, 2),
        ("at_least", ["Coi"], False, 0.3, 1),
        ("at_least", ["Cong tre"], False, 0.3, 1),
        ("at_least", ["Den nuoc"], False, 0.3, 2),
        ("at_least", ["Den troi"], False, 0.5, 15),
        ("at_least", ["Cau con ong"], False, 0.3, 1)
    ],

    "Tết Choi Chnam Thmay": [
        ("confidence_min", ["Nui cat"], True, 1.0, 0.8),
        ("confidence_min", ["Choi Chnam Thmay"], True, 1.0, 0.9),
        ("at_least_in_frame", ["Nguoi mua", "Nguoi mua chan", "trong Chhay-dam"], False, 0.6, None),
        ("at_least_in_frame", ["Tuong Phat", "Co Phat dan", "Chua", "Nuoc thom"], False, 0.8, None),
        ("at_least_in_frame", ["Nguoi tham gia te nuoc", "Nuoc thom", "Tuong Phat"], False, 0.6, None),
        ("at_least_in_frame", ["Nuoc thom", "Nha su", "Tuong Phat"], False, 0.6, None),
        ("is_presence_in_frame", ["Than 3 mat", "Du ruoc doan"], True, 1.0, None),
        ("is_presence_in_frame", ["Than 3 mat", "Nguoi mua chan"], True, 1.0, None),
        ("is_presence_in_frame", ["Banh gung", "Banh num kha mos"], True, 1.0, None)
    ]
}

# Map subclass → possible lễ_hội (multi-map)
SUBCLASS_TO_FESTIVAL = {
    # Tết Choi Chnam Thmay
    "Chua": ["Tết Choi Chnam Thmay"],
    "Co Phat dan": ["Tết Choi Chnam Thmay"],
    "Tuong Phat": ["Tết Choi Chnam Thmay"],
    "Nuoc thom": ["Tết Choi Chnam Thmay"],
    "Nguoi tham gia te nuoc": ["Tết Choi Chnam Thmay"],
    "Nguoi mua chan": ["Tết Choi Chnam Thmay"],
    "Nha su": ["Tết Choi Chnam Thmay"],
    "Nui cat": ["Tết Choi Chnam Thmay"],
    "Nguoi mua": ["Tết Choi Chnam Thmay"],
    "Choi Chnam Thmay": ["Tết Choi Chnam Thmay"],
    "trong Chhay-dam": ["Tết Choi Chnam Thmay"],
    "Banh gung": ["Tết Choi Chnam Thmay"],
    "Than 3 mat": ["Tết Choi Chnam Thmay"],
    "Du ruoc doan": ["Tết Choi Chnam Thmay"],
    "Banh num kha mos": ["Tết Choi Chnam Thmay"],

    # Ooc Bom Bóc
    "Chay": ["Ooc Bom Bóc"],
    "Coi": ["Ooc Bom Bóc"],
    "Ghe ngo": ["Ooc Bom Bóc"],
    "Nguoi mua": ["Ooc Bom Bóc"],
    "Den nuoc": ["Ooc Bom Bóc"],
    "Den troi": ["Ooc Bom Bóc"],
    "Den hoa dang": ["Ooc Bom Bóc"],
    "Com": ["Ooc Bom Bóc"],
    "Khoai": ["Ooc Bom Bóc"],
    "Den hoa dang dang thung": ["Ooc Bom Bóc"],
    "trong Chhay-dam": ["Ooc Bom Bóc"],
    "Cong tre": ["Ooc Bom Bóc"],
    "Cau con ong": ["Ooc Bom Bóc"],
    "Ok bom boc": ["Ooc Bom Bóc"],

    # Chợ nổi Cái Răng
    "Cay beo": ["Chợ nổi Cái Răng"],
    "Thuyen": ["Chợ nổi Cái Răng"],

    # Lễ hội thác côn
    "binh bong dua": ["Lễ hội thác côn"],
    "hoa sen": ["Lễ hội thác côn"],
    "nha su": ["Lễ hội thác côn"],

    # Nhạc Ngũ Âm người Khmer
    "Ken Srolay pin-piet": ["Nhạc Ngũ Âm người Khmer"],
    "Dan sat Ro-niet-dek": ["Nhạc Ngũ Âm người Khmer"],
    "Dan cong Coung-touch Coung-thom": ["Nhạc Ngũ Âm người Khmer"],
    "Dan thuyen Ro-niet-ek": ["Nhạc Ngũ Âm người Khmer"],
    "Trong lon Sakho-thom": ["Nhạc Ngũ Âm người Khmer"],
    "Trong nho Kha-so-sompho": ["Nhạc Ngũ Âm người Khmer"],
    "Trang phuc Khmer": ["Nhạc Ngũ Âm người Khmer"],

    # Lễ hội Kỳ Yên Đình Bình Thủy
    "Nghe si Hat Boi": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "San khau Hat Boi": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Mua Lan": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Mieu": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Ban tho Dinh Binh Thuy": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Ao dai": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Le phuc Ky Yen": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Le vat Ky Yen": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Xoi": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Dinh Binh Thuy": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Cong Dinh": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Xe dieu hanh": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Khan Sac Than": ["Lễ hội Kỳ Yên Đình Bình Thủy"],
    "Sac Than": ["Lễ hội Kỳ Yên Đình Bình Thủy"],

    # Đờn ca tài tử
    "dan kim": ["Đờn ca tài tử"],
    "dan guitar": ["Đờn ca tài tử"],
    "dan bau": ["Đờn ca tài tử"],
    "dan co": ["Đờn ca tài tử"],
    "dan ty ba": ["Đờn ca tài tử"],
    "dan tam": ["Đờn ca tài tử"],
    "sao": ["Đờn ca tài tử"],
    "dan tranh": ["Đờn ca tài tử"],

    # Sân khấu Dù Kê
    "Giap nguc dinh hat": ["Sân Khấu Dù Kê"],
    "Tay ao dinh hat": ["Sân Khấu Dù Kê"],
    "Toc bui cao": ["Sân Khấu Dù Kê"],
    "Deo vong tay": ["Sân Khấu Dù Kê"],
    "Dau doi khan dong": ["Sân Khấu Dù Kê"],
    "Ao ren dinh hat": ["Sân Khấu Dù Kê"],
    "Vay sampot": ["Sân Khấu Dù Kê"],
    "That lung ban to": ["Sân Khấu Dù Kê"],
    "Deo vong co": ["Sân Khấu Dù Kê"],
    "Do vai bong": ["Sân Khấu Dù Kê"],
    "Giap than duoi": ["Sân Khấu Dù Kê"],
    "Canh thien nhien trong rung": ["Sân Khấu Dù Kê"],
    "Long may ke dam": ["Sân Khấu Dù Kê"],
    "Moi do": ["Sân Khấu Dù Kê"],
    "Mac do dinh hat": ["Sân Khấu Dù Kê"],
    "Khoac ao ren": ["Sân Khấu Dù Kê"],
    "Trang diem mat quy": ["Sân Khấu Dù Kê"],
    "Canh cung dien": ["Sân Khấu Dù Kê"],
    "Canh hang dong": ["Sân Khấu Dù Kê"],
    "Canh bien": ["Sân Khấu Dù Kê"],
    "Khan ran": ["Sân Khấu Dù Kê"],
    "Cam vu khi": ["Sân Khấu Dù Kê"],
    "Ao khoac vai": ["Sân Khấu Dù Kê"],
    "Rau dai": ["Sân Khấu Dù Kê"],
    "Nguoi bieu dien": ["Sân Khấu Dù Kê"],
    "Cam gay": ["Sân Khấu Dù Kê"],

    # Nginh Ông
    "Ao dai truyen thong": ["Nginh Ông"],
    "Lan su rong": ["Nginh Ông"],
    "Trang phuc le hoi": ["Nginh Ông"],
    "Trang phuc bieu dien nghe thuat nhan gian": ["Nginh Ông"],
    "Trang phuc linh": ["Nginh Ông"],
    "Ban tho ong": ["Nginh Ông"],
    "Tau bien": ["Nginh Ông"],
    "Kieu ruoc ong": ["Nginh Ông"],
    "Trong": ["Nginh Ông"],
    "Xe dieu hanh ruoc ong": ["Nginh Ông"],
    "Cong dinh ong": ["Nginh Ông"]
}
